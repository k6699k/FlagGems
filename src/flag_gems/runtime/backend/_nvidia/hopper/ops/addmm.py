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

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.ops.addmm import _addmm_impl
from flag_gems.ops.copy import copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to

from .mm import (
    _CLUSTER_MESH,
    TensorDescriptor,
    _addmm_batch_tma,
    _addmm_epilogue,
    _addmm_splitk,
    _addmm_tma,
    _batched_mm_tma_grid,
    _packed_mm_splitk_grid,
    _packed_mm_tma_grid,
)


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
    # Reduce coalesced row-major B loads along K, keeping adjacent output
    # columns next to one another in memory.
    if ROW_VECTOR and SB1 == 1 and K <= BK:
        vec = tl.load(A + rows * SA0 + kk * SA1, kk < K, 0).to(tl.float32)
        mat = tl.load(
            B + kk[:, None] * SB0 + cols[None, :],
            (kk[:, None] < K) & (cols[None, :] < N),
            0,
        )
        acc = tl.sum(mat.to(tl.float32) * vec[:, None], axis=0)
    else:
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
        _addmm_epilogue(acc, bias, alpha, beta, ALPHA_ONE, BETA),
        valid,
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
    tl.store(
        Out + offs, _addmm_epilogue(acc, bias, alpha, beta, ALPHA_ONE, BETA), valid
    )


@lru_cache(maxsize=1024)
def _narrow_config(m, n, k):
    if not (16 <= m <= 8192 and 16 <= n <= 512 and k >= 1024 and k % 256 == 0):
        return None
    # Map the 64-row WGMMA dimension onto the output columns. When these
    # tiles underfill the SMs, CTAs in one cluster share the K reduction.
    if m > 2048 and n > 128 or m > 512 and n > 384:
        return 64, 128, 128, 1, False
    if m > 512:
        bm = 16 if m <= 1536 and n <= 128 else 32
    else:
        bm = (
            32
            if k >= 4096 and (n > 256 or (n >= 128 and k >= 6144 and m < 384))
            else 16
        )
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, 64)
    splits = 1
    if k >= 4096 and tiles <= 32:
        splits = 4
    elif k >= 2048 and tiles <= 64:
        splits = 2
    splits = min(splits, (k // 256) & -(k // 256))
    return bm, 64, 256, splits if splits in _CLUSTER_MESH else 1, True


@lru_cache(maxsize=1024)
def _tma_config(m, n, k, device):
    if m <= 64 and n >= 16384 and k >= 1024:
        # Large weights dominate the traffic. Use wide contiguous TMA loads
        # and fit short M dimensions inside one WGMMA row tile.
        return 64, 128, 64, 4, 4, False
    if m < 16 and n >= 2048 and k >= 512:
        return 16, 64, 128, 4, 4, True
    if 16 <= m <= 512 and n >= 1024:
        # Let the grid fill a wave before expanding either tile dimension.
        # A larger K tile is profitable when a single wave owns the full grid.
        tiles = triton.cdiv(m, 64) * triton.cdiv(n, 128)
        if k <= 256:
            if m <= 128:
                return 16, 128, min(64, triton.next_power_of_2(k)), 3, 4, True
            return 64, 128, 64, 3, 4, False
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        if n >= 8192 and m >= 128 and k >= 1024:
            large_tiles = triton.cdiv(m, 128) * triton.cdiv(n, 256)
            if large_tiles >= sm_count // 2:
                return 128, 256, 64, 3, 8, False
        if n <= 8192 and sm_count // 2 <= tiles <= sm_count and k >= 1536:
            return 64, 128, 128, 4, 4, False
        if n <= 4096 and tiles > sm_count and k >= 1024:
            return 64, 128, 64, 4, 4, False
        if 64 <= triton.cdiv(m, 64) * triton.cdiv(n, 64) < 128 and k >= 1536:
            return 64, 64, 128, 4, 4, False
    # For short/wide outputs, swap the dot operands so the hardware's 64-row
    # WGMMA tile runs along N. More, narrower M tiles expose more parallelism.
    if 16 <= m <= 512 and n >= 128 and k >= 512 and m * n <= 1048576:
        bm = 16 if n <= 512 and k >= 2048 else 32
        bk = 128 if n > 512 and k <= 1024 else 256
        stages = 4 if k >= 2048 else 3
        return bm, 64, bk, stages, 4, True
    if n <= 64 or (m <= 512 and n <= 512):
        return 64, 64, 128, 4, 4, False
    if m >= 1024:
        return 128, 256, 64, 3, 8, False
    if 64 < m <= 128 and k >= 4096:
        # One M tile, so doubling the N tile can fit the whole grid in one
        # wave. Avoid a partly occupied second wave of long-lived CTAs.
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        if sm_count < triton.cdiv(n, 128) <= 2 * sm_count:
            return 128, 256, 64, 3, 8, False
    return 128, 128, 64, 4, 4, False


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    if mat2.device != mat1.device or bias.device != mat1.device:
        raise RuntimeError("addmm expects all operands to be on the same device")
    device = mat1.device
    if device.type != "cuda":
        return _addmm_impl(bias, mat1, mat2, None, beta, alpha)
    if device.index != torch_device_fn.current_device():
        with torch_device_fn.device(device):
            return addmm(bias, mat1, mat2, beta=beta, alpha=alpha)
    m, k = mat1.shape
    kb, n = mat2.shape
    matrix_bias = bias.shape == (m, n)
    if (
        alpha == 0
        and beta in (0, 1)
        and kb == k
        and mat1.dtype == mat2.dtype == bias.dtype
        and mat1.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        and (matrix_bias or broadcastable_to(bias.shape, (m, n)))
    ):
        if beta == 0:
            return torch.zeros((m, n), device=device, dtype=mat1.dtype)
        return copy_(torch.empty((m, n), device=device, dtype=mat1.dtype), bias)
    row_vec = m <= 2
    wide = m < 16 and n >= 2048 and k >= 512
    vector = row_vec and not wide or n == 1
    # Split long reductions whose output tiles alone underfill the GPU.
    # Use software partials for matrices narrower than a tensor-core tile.
    small_output = m * n <= 32768 and k >= 1024
    long_reduction = m * n <= 131072 and k >= 4096
    split_k = (
        m <= 256
        and (small_output or long_reduction)
        or 256 < m <= 8192
        and n < 16
        and k >= 1024
    ) and not (vector or wide)
    narrow = (
        _narrow_config(m, n, k) if hasattr(TensorDescriptor, "from_tensor") else None
    )
    tma = (
        hasattr(TensorDescriptor, "from_tensor")
        and (m >= 16 or wide)
        and n >= 64
        and k >= 64
        and triton.cdiv(m, 16) * triton.cdiv(n, 64) >= 32
    )
    if tma and not small_output and m * n >= 65536:
        split_k = False
    sb0, sb1 = mat2.stride()
    trans_b = sb1 != 1
    if (tma or narrow) and (
        k % 8 or (not trans_b and n % 8) or mat1.data_ptr() % 16 or mat2.data_ptr() % 16
    ):
        narrow = None
        tma = False
    if (
        not (vector or split_k or tma or narrow)
        or min(m, n, k) == 0
        or k != kb
        or mat1.dtype not in (torch.float16, torch.bfloat16)
        or mat2.dtype != mat1.dtype
        or bias.dtype != mat1.dtype
        or (not matrix_bias and not broadcastable_to(bias.shape, (m, n)))
        or max(m * k, k * n, m * n) >= 2**31
        or not mat1.is_contiguous()
        or not (
            (sb1 == 1 and (sb0 == n or k == 1)) or (sb0 == 1 and (sb1 == k or n == 1))
        )
    ):
        return _addmm_impl(bias, mat1, mat2, None, beta, alpha)
    out = torch.empty((m, n), device=mat1.device, dtype=mat1.dtype)
    # Views dispatched through ATen add host cost at decode sizes. Derive
    # broadcast strides directly, including zero-stride expanded inputs.
    si = (
        bias.stride()
        if matrix_bias
        else (
            bias.stride(0) if bias.ndim == 2 and bias.shape[0] != 1 else 0,
            bias.stride(-1) if bias.ndim and bias.shape[-1] != 1 else 0,
        )
    )
    beta_kind = 0 if beta == 0 else 1 if beta == 1 else 2
    if vector:
        # Warp-parallel reduction of a full K row is inexpensive for
        # decode; the 1024-thread CTA shortens very long reductions.
        if k <= 16384 and (row_vec or k & (k - 1) == 0):
            tile = 4 if row_vec and not trans_b and n % 4 == 0 and n >= 32 else 1
            if row_vec and trans_b and k <= 256 and n >= 512:
                # Reuse a short input row across contiguous rows of the weight.
                tile = 32
            bk = triton.next_power_of_2(k)
            warps = 32 if bk >= 8192 else (8 if row_vec else 4)
        else:
            tile, bk, warps = 4, min(triton.next_power_of_2(k), 512), 4
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
            num_warps=warps,
        )
    elif narrow:
        bm, bn, bk, splits, trans_dot = narrow
        ad = TensorDescriptor(mat1, (m, k), (k, 1), [bm, bk])
        bd = TensorDescriptor(
            mat2,
            (n, k) if trans_b else (k, n),
            (k if trans_b else n, 1),
            [bn, bk] if trans_b else [bk, bn],
        )
        _addmm_batch_tma[_batched_mm_tma_grid](
            ad,
            bd,
            out,
            bias,
            alpha,
            beta,
            0,
            *si,
            True,
            beta_kind != 0,
            m,
            n,
            k,
            1,
            bm,
            bn,
            bk,
            splits,
            trans_b,
            trans_dot,
            _CLUSTER_MESH.get(splits),
            PLAN_STAGES=4,
            PLAN_WARPS=4,
        )
    elif split_k:
        splits = 16 if small_output or n < 16 else 4
        if n < 16 and k >= 8192:
            splits = 32
        bm, bn, bk = 16, 16 if n < 16 else 32, 64
        partials = torch.empty((splits, m, n), device=mat1.device, dtype=torch.float32)
        _addmm_splitk[_packed_mm_splitk_grid](
            mat1,
            mat2,
            partials,
            m,
            n,
            k,
            *mat1.stride(),
            *mat2.stride(),
            partials.stride(1),
            partials.stride(2),
            partials.stride(0),
            splits,
            bm,
            bn,
            bk,
            PLAN_STAGES=3,
            PLAN_WARPS=4,
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
        bm, bn, bk, stages, warps, trans_dot = _tma_config(m, n, k, mat1.device)
        a = TensorDescriptor.from_tensor(mat1, [bm, bk])
        b = TensorDescriptor(
            mat2,
            (n, k) if trans_b else (k, n),
            (sb1, sb0) if trans_b else (sb0, sb1),
            [bn, bk] if trans_b else [bk, bn],
        )
        _addmm_tma[_packed_mm_tma_grid](
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
            TRANS_DOT=trans_dot,
            PLAN_STAGES=stages,
            PLAN_WARPS=warps,
        )
    return out
