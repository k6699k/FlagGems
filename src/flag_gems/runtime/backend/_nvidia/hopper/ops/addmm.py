# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

from flag_gems.ops.addmm import _addmm_impl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None


@triton.jit
def _combine(acc, bias, alpha, beta, ALPHA_ONE: tl.constexpr, BETA: tl.constexpr):
    if not ALPHA_ONE:
        acc = acc * alpha
    if BETA == 1:
        acc = acc + bias
    elif BETA != 0:
        acc = acc + beta * bias
    return acc


@triton.jit
def _addmm_vector(
    A,
    B,
    Bias,
    Out,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SA0: tl.constexpr,
    SA1: tl.constexpr,
    SB0: tl.constexpr,
    SB1: tl.constexpr,
    SI0: tl.constexpr,
    SI1: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    TILE: tl.constexpr,
    BK: tl.constexpr,
    ROW_VECTOR: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA: tl.constexpr,
):
    # Keep the reduction inside a CTA for decode and narrow projections.
    pos = tl.program_id(0) * TILE + tl.arange(0, TILE)
    single = tl.program_id(1)
    if ROW_VECTOR:
        rows, cols = single, pos
    else:
        rows, cols = pos, single
    kk = tl.arange(0, BK)
    products = tl.full((TILE, BK), 0, tl.float32)
    for i in range(tl.cdiv(K, BK)):
        k = i * BK + kk
        if ROW_VECTOR:
            vec = tl.load(A + rows * SA0 + k * SA1, k < K, 0).to(tl.float32)
            mat = tl.load(
                B + k[None, :] * SB0 + cols[:, None] * SB1,
                (cols[:, None] < N) & (k[None, :] < K),
                0,
            )
        else:
            vec = tl.load(B + k * SB0 + cols * SB1, k < K, 0).to(tl.float32)
            mat = tl.load(
                A + rows[:, None] * SA0 + k[None, :] * SA1,
                (rows[:, None] < M) & (k[None, :] < K),
                0,
            )
        products = tl.fma(vec[None, :], mat.to(tl.float32), products)
    acc = tl.sum(products, axis=1)
    valid = (rows < M) & (cols < N)
    rows = rows.to(tl.int64)
    cols = cols.to(tl.int64)
    if BETA != 0:
        bias = tl.load(Bias + rows * SI0 + cols * SI1, valid, 0).to(tl.float32)
    else:
        bias = 0.0
    tl.store(
        Out + rows * SO0 + cols * SO1,
        _combine(acc, bias, alpha, beta, ALPHA_ONE, BETA),
        valid,
    )


@triton.jit
def _addmm_splitk(
    A,
    B,
    Partials,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SB0: tl.constexpr,
    SB1: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Each CTA handles consecutive K tiles. FP32 partials retain the full
    # accumulator precision until the bias and scaling are applied.
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    tiles = tl.cdiv(K, SPLITS * BK)
    start = split * tiles * BK + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(tiles):
        kk = start + i * BK
        a = tl.load(
            A + rows[:, None] * K + kk[None, :],
            (rows[:, None] < M) & (kk[None, :] < K),
            0.0,
        )
        b = tl.load(
            B + kk[:, None] * SB0 + cols[None, :] * SB1,
            (kk[:, None] < K) & (cols[None, :] < N),
            0.0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        Partials + split * M * N + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _addmm_finish(
    Partials,
    Bias,
    Out,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    SI0: tl.constexpr,
    SI1: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < M * N
    acc = tl.sum(
        tl.load(
            Partials + tl.arange(0, SPLITS)[:, None] * M * N + offs[None, :],
            valid[None, :],
            0.0,
        ),
        0,
    )
    if BETA != 0:
        rows = (offs // N).to(tl.int64)
        cols = (offs % N).to(tl.int64)
        bias = tl.load(Bias + rows * SI0 + cols * SI1, valid, 0).to(tl.float32)
    else:
        bias = 0.0
    tl.store(Out + offs, _combine(acc, bias, alpha, beta, ALPHA_ONE, BETA), valid)


@triton.jit
def _addmm_tma(
    A,
    B,
    Bias,
    Out,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SI0: tl.constexpr,
    SI1: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TRANS_B: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA: tl.constexpr,
):
    pid = tl.program_id(0)
    # Group adjacent row tiles to reuse the same B tile from L2.
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = pid // (8 * num_n)
    first_m = group * 8
    group_m = tl.minimum(num_m - first_m, 8)
    in_group = pid % (8 * num_n)
    m = (first_m + in_group % group_m) * BM
    n = (in_group // group_m) * BN
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(tl.cdiv(K, BK)):
        a = A.load([m, i * BK])
        if TRANS_B:
            b = tl.trans(B.load([n, i * BK]))
        else:
            b = B.load([i * BK, n])
        acc = tl.dot(a, b, acc)
    rows = (m + tl.arange(0, BM)).to(tl.int64)
    cols = (n + tl.arange(0, BN)).to(tl.int64)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    if BETA != 0:
        bias = tl.load(Bias + rows[:, None] * SI0 + cols[None, :] * SI1, valid, 0).to(
            tl.float32
        )
    else:
        bias = 0.0
    tl.store(
        Out + rows[:, None] * SO0 + cols[None, :] * SO1,
        _combine(acc, bias, alpha, beta, ALPHA_ONE, BETA),
        valid,
    )


def _tma_config(m, n, k, device, trans_b):
    if n <= 64 or (m <= 512 and n <= 512):
        return 64, 64, 128, 4, 4
    if m >= 1024:
        return 128, 256, 64, 3, 8
    if 64 < m <= 128 and not trans_b and k >= 4096:
        # One M tile, so doubling the N tile can fit the whole grid in one
        # wave. Avoid a partly occupied second wave of long-lived CTAs.
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        if sm_count < triton.cdiv(n, 128) <= 2 * sm_count:
            return 128, 256, 64, 3, 8
    return 128, 128, 64, 4, 4


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    if mat2.device != mat1.device or bias.device != mat1.device:
        raise RuntimeError("addmm expects all operands to be on the same device")
    m, k = mat1.shape
    kb, n = mat2.shape
    row_vec = m <= 2
    vector = row_vec or n == 1
    # Split long reductions whose output tiles alone underfill the GPU.
    # Both regimes cap the FP32 workspace at 2 MiB.
    small_output = m * n <= 32768 and k >= 1024
    long_reduction = m * n <= 131072 and k >= 4096
    split_k = m <= 256 and (small_output or long_reduction) and not vector
    tma = (
        hasattr(TensorDescriptor, "from_tensor")
        and m >= 64
        and n >= 64
        and k >= 64
        and triton.cdiv(m, 64) * triton.cdiv(n, 128) >= 32
    )
    if not (vector or split_k or tma):
        return _addmm_impl(bias, mat1, mat2, None, beta, alpha)
    trans_b = mat2.stride(1) != 1
    if (
        min(m, n, k) == 0
        or k != kb
        or mat1.dtype not in (torch.float16, torch.bfloat16)
        or mat2.dtype != mat1.dtype
        or bias.dtype != mat1.dtype
        or not broadcastable_to(bias.shape, (m, n))
        or max(m * k, k * n, m * n) >= 2**31
        or not mat1.is_contiguous()
        or not (mat2.t() if trans_b else mat2).is_contiguous()
        or (
            tma
            and not (vector or split_k)
            and (
                k % 8
                or (not trans_b and n % 8)
                or mat1.data_ptr() % 16
                or mat2.data_ptr() % 16
            )
        )
    ):
        return _addmm_impl(bias, mat1, mat2, None, beta, alpha)
    out = torch.empty((m, n), device=mat1.device, dtype=mat1.dtype)
    si = bias.expand(m, n).stride()
    beta_kind = 0 if beta == 0 else 1 if beta == 1 else 2
    with torch_device_fn.device(mat1.device):
        if vector:
            tile = 4 if row_vec else 8
            bk = min(triton.next_power_of_2(k), 512)
            grid = (triton.cdiv(n, tile), m) if row_vec else (triton.cdiv(m, tile), n)
            _addmm_vector[grid](
                mat1,
                mat2,
                bias,
                out,
                alpha,
                beta,
                m,
                n,
                k,
                *mat1.stride(),
                *mat2.stride(),
                *si,
                *out.stride(),
                tile,
                bk,
                row_vec,
                alpha == 1,
                beta_kind,
                num_warps=4,
            )
        elif split_k:
            splits = 16 if small_output else 4
            bm, bn, bk = 16, 32, 64
            partials = torch.empty(
                (splits, m, n), device=mat1.device, dtype=torch.float32
            )
            _addmm_splitk[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
                mat1,
                mat2,
                partials,
                m,
                n,
                k,
                *mat2.stride(),
                splits,
                bm,
                bn,
                bk,
            )
            _addmm_finish[(triton.cdiv(m * n, 128),)](
                partials,
                bias,
                out,
                alpha,
                beta,
                m,
                n,
                *si,
                splits,
                128,
                alpha == 1,
                beta_kind,
            )
        else:
            bm, bn, bk, stages, warps = _tma_config(m, n, k, mat1.device, trans_b)
            a = TensorDescriptor.from_tensor(mat1, [bm, bk])
            b = TensorDescriptor.from_tensor(
                mat2.t() if trans_b else mat2, [bn, bk] if trans_b else [bk, bn]
            )
            _addmm_tma[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
                a,
                b,
                bias,
                out,
                alpha,
                beta,
                m,
                n,
                k,
                *si,
                *out.stride(),
                bm,
                bn,
                bk,
                trans_b,
                alpha == 1,
                beta_kind,
                num_stages=stages,
                num_warps=warps,
            )
    return out
