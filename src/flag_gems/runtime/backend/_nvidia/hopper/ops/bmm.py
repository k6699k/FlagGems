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

from flag_gems.ops.bmm import bmm as generic_bmm
from flag_gems.runtime import flagtune_enabled, torch_device_fn

from .mm import (
    _CLUSTER_MESH,
    TensorDescriptor,
    _baddbmm_tma,
    _batch_mm,
    _batched_mm_tma_grid,
    _bmm_tma,
    _store_batched_mm_result,
)


@triton.jit
def _batch_gemv(
    matrix,
    vector,
    bias,
    out,
    rows: tl.constexpr,
    K: tl.constexpr,
    xm: tl.constexpr,
    xk: tl.constexpr,
    xb: tl.constexpr,
    vr: tl.constexpr,
    vk: tl.constexpr,
    vb: tl.constexpr,
    ym: tl.constexpr,
    yv: tl.constexpr,
    yb: tl.constexpr,
    cm: tl.constexpr,
    cv: tl.constexpr,
    cb: tl.constexpr,
    alpha,
    beta,
    HAS_BIAS: tl.constexpr,
    READ_BIAS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    v = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    k = tl.arange(0, BLOCK_K)
    # Each stride product is evaluated before addition to the 64-bit base.
    if rows * xm >= 2**31 or rows * ym >= 2**31 or rows * cm >= 2**31:
        row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    if BLOCK_K * xk >= 2**31 or BLOCK_K * vk >= 2**31:
        k = k.to(tl.int64)
    a = tl.load(
        matrix + batch * xb + row[:, None] * xm + k[None, :] * xk,
        (row[:, None] < rows) & (k[None, :] < K),
        other=0,
    )
    b = tl.load(vector + batch * vb + v * vr + k * vk, k < K, other=0)
    acc = tl.sum(a.to(tl.float32) * b[None, :].to(tl.float32), axis=1)
    _store_batched_mm_result(
        acc,
        bias,
        out,
        batch * yb + row * ym + v * yv,
        row < rows,
        batch * cb + row * cm + v * cv,
        alpha,
        beta,
        HAS_BIAS,
        READ_BIAS,
    )


def _use_narrow_cluster(m, n, k):
    return (
        bool(_CLUSTER_MESH)
        and 2048 <= k <= 8192
        and k % 256 == 0
        and (
            (8 <= m <= 256 and n == 16 and (k >= 4096 or m > 32))
            or (8 <= m <= 64 and 16 < n <= 64 and k >= 4096)
        )
    )


def _can_tma(A, B, batch, m, n, k):
    # Fold packed batches into the leading dimension of a 2-D tensor map.
    # A matrix tail may then load unused rows from the next batch. These
    # lanes cannot affect valid dot products and are masked at the store.
    return (
        TensorDescriptor is not None
        and 1 <= batch <= 65535
        and (
            m >= 8
            and m * n >= 4096
            or 0 < m <= 4
            and n >= 2048
            and (m > 1 or n >= 4096 or B.stride(1) != 1)
            or _use_narrow_cluster(m, n, k)
        )
        and 16 <= n
        and (k >= 512 or (128 <= k <= 256 and 128 <= m <= 512 and n >= 512))
        and k % 64 == 0
        and max(batch * m, batch * n, batch * k) < 2**31
        and A.stride() == (m * k, k, 1)
        and (
            B.stride() == (k * n, 1, k) or (n % 8 == 0 and B.stride() == (k * n, n, 1))
        )
        and A.data_ptr() % 16 == 0
        and B.data_ptr() % 16 == 0
    )


@lru_cache(maxsize=1024)
def _tma_plan(batch, m, n, k, nt):
    keep_small_tuning = (
        batch <= 16
        and 32 <= m <= 512
        and m % 32 == 0
        and 128 <= n <= 512
        and n % 64 == 0
        and 1024 <= k <= 8192
        and k % 128 == 0
        and (m >= 64 or bool(_CLUSTER_MESH))
    )
    warps = 4
    if _use_narrow_cluster(m, n, k):
        # A narrow output grid cannot occupy the device without K partitions.
        bm, bn = (64, 16) if n == 16 and m >= 96 else (32 if m >= 128 else 16, 64)
        bk, stages, trans_dot = 256, 3, bm < 64
        blocks = k // bk
        max_splits = 4 if batch * triton.cdiv(m, bm) < 64 else 2
        splits = min(max_splits, blocks & -blocks)
    elif keep_small_tuning and m < 64:
        bm, bn, bk = (16 if batch <= 2 or (batch <= 4 and nt) else 32), 64, 128
        trans_dot = True
        blocks = k // bk
        splits = min(2 if batch >= 8 and not nt else 4, blocks & -blocks)
        stages = 3
        if nt and batch >= 4 and k % 512 == 0:
            # Two deeper K partitions keep transposed loads contiguous.
            bm, bk, splits, stages = 32, 256, 2, 4
    elif keep_small_tuning:
        # As the batch grows the same tile grid spans more Hopper SMs.
        # Widen tiles once there is enough parallel work to cover the device.
        tile_count = batch * (m // 64) * (n // 64)
        if m % 64 == 0 and tile_count >= 256 and n % 128 == 0:
            bm, bn, bk, stages, trans_dot = 64, 128, 128, 4, nt
        elif m % 64 == 0 and tile_count >= 128:
            bm, bn, bk, stages, trans_dot = 64, 64, 128, 5, nt
        else:
            bm, bn = (64, 32) if m % 64 == 0 else (32, 64)
            bk = 256 if k % 256 == 0 else 128
            stages, trans_dot = 4, bm < 64
        splits = 1
    else:
        bm, bn, bk, stages, warps, trans_dot = _tma_tile(batch, m, n, k)
        splits = 1
        tiles = batch * triton.cdiv(m, bm) * triton.cdiv(n, bn)
        medium_projection = 128 <= m <= 512 and 512 <= n <= 4096
        if _CLUSTER_MESH and tiles < 64 and k >= 1024 and not medium_projection:
            blocks = k // bk
            splits = min(4, blocks & -blocks)
    return bm, bn, bk, splits, trans_dot, stages, warps


def _launch_tma(A, B, out, bias, alpha, beta, bias_strides, has_bias):
    batch, m, k = A.shape
    n = B.size(2)
    nt = B.stride(1) == 1
    bm, bn, bk, splits, trans_dot, stages, warps = _tma_plan(batch, m, n, k, nt)
    ad = TensorDescriptor(A, [batch * m, k], [k, 1], [bm, bk])
    width = k if nt else n
    height = n if nt else k
    bd = TensorDescriptor(
        B, [batch * height, width], [width, 1], [bn, bk] if nt else [bk, bn]
    )
    read_bias = has_bias and beta != 0
    kernel = _baddbmm_tma if has_bias else _bmm_tma
    kernel[_batched_mm_tma_grid](
        ad,
        bd,
        out,
        bias,
        alpha,
        beta,
        *bias_strides,
        has_bias,
        read_bias,
        m,
        n,
        k,
        batch,
        bm,
        bn,
        bk,
        splits,
        nt,
        trans_dot,
        _CLUSTER_MESH.get(splits),
        PLAN_STAGES=stages,
        PLAN_WARPS=warps,
    )


def _tma_tile(batch, m, n, k):
    # Use short M tiles for decode; broader N tiles raise arithmetic intensity
    # once the launch already has a full wave of independent output tiles.
    if k <= 256:
        # Short K leaves at most four iterations. Use a shallow pipeline
        # and only widen the output tiles when the grid still fills the GPU.
        if m >= 320 and batch * triton.cdiv(m, 128) * triton.cdiv(n, 128) >= 128:
            return 128, 128, 64, 2, 4, False
        wide = batch * triton.cdiv(m, 64) * triton.cdiv(n, 128) >= 128
        return 64, 128 if wide else 64, 64, 2, 4, False
    if 128 <= m <= 512 and 512 <= n <= 4096:
        bk = 128 if k % 128 == 0 else 64
        tiles_128 = batch * triton.cdiv(m, 128) * triton.cdiv(n, 128)
        if m >= 352 and n >= 2048 and tiles_128 >= 64:
            return 128, 128, bk, 3, 4, False
        if n >= 1024 and batch * triton.cdiv(m, 64) * triton.cdiv(n, 128) >= 64:
            return 64, 128, bk, 4, 4, False
        return 64, 64, bk, 5, 4, False
    if (
        128 < m < 384
        and n > 4096
        and batch * triton.cdiv(m, 128) * triton.cdiv(n, 256) >= 128
    ):
        return 128, 256, 64, 3, 8, False
    if m <= 4:
        if n < 4096:
            bk = 256 if k % 256 == 0 else (128 if k % 128 == 0 else 64)
            return 16, 64, bk, 4, 4, True
        return 16, 256, (128 if k % 128 == 0 else 64), 3, 4, True
    if m <= 128 and n >= 8192:
        bm = max(16, triton.next_power_of_2(m))
        bk = 128 if k % 128 == 0 else 64
        return bm, 128, bk, 3, 4, bm < 64
    if m < 48:
        bm = 16 if m <= 16 else 32
        bn = 128 if n >= 2048 else 64
        return bm, bn, (128 if k % 128 == 0 else 64), 3, 4, True
    if n >= 512 and batch * triton.cdiv(m, 128) * triton.cdiv(n, 256) >= 128:
        if m >= 384:
            return 128, 256, 64, 3, 8, False
        if m >= 96 and k % 128 == 0:
            return 128, 128, 128, 3, 4, False
        if m >= 64:
            return 64, 128, 64, 4, 4, False
    bm = 32 if m < 64 else 64
    bk = 128 if k % 128 == 0 else 64
    return bm, 64, bk, 4, 4, bm < 64


def _batch_path(A, B, *, fused_bias=False, ignore_bias=False):
    if A.ndim != 3 or B.ndim != 3:
        raise RuntimeError("bmm expects two rank-three tensors")
    batch, m, k = A.shape
    b, other_k, n = B.shape
    if batch != b or k != other_k:
        raise RuntimeError("bmm expects matching batch and contracting dimensions")
    if A.dtype != B.dtype or A.device != B.device:
        raise RuntimeError(
            "bmm expects both inputs on the same device with the same dtype"
        )
    if batch == 0 or batch > 65535 or min(m, n, k) == 0:
        return None
    if A.dtype not in (torch.float16, torch.bfloat16):
        return None

    gemv = min(m, n) <= 4 and 256 <= k <= 16384
    if ignore_bias and n <= 16 * 65535:
        if _can_tma(A, B, batch, m, n, k):
            return "tma"
        return "gemv" if gemv else "mma"
    if A.stride(2) != 1 or min(B.stride()[1:]) != 1:
        return None

    compact = (
        m <= 512
        and n <= 512
        and (1024 <= k <= 8192 or (16 <= k <= 128 and min(m, n) >= 16))
    )
    if _can_tma(A, B, batch, m, n, k):
        return "tma"
    if not (gemv or compact) and flagtune_enabled("baddbmm" if fused_bias else "bmm"):
        return None
    if gemv:
        return "gemv"
    if compact or (fused_bias and B.stride(1) == 1 and 4 < n <= 16 * 65535 and m > 4):
        return "mma"
    return None


def _batch_product(A, B, path, bias=None, alpha=1, beta=0):
    if A.device.index != torch_device_fn.current_device():
        with torch_device_fn.device(A.device):
            return _batch_product(A, B, path, bias, alpha, beta)

    batch, m, k = A.shape
    n = B.shape[2]
    out = torch.empty((batch, m, n), device=A.device, dtype=A.dtype)
    has_bias = bias is not None
    if has_bias:
        # Integer one lets Triton fold the unit scales out of the epilogue.
        alpha = 1 if alpha == 1 else alpha
        beta = 1 if beta == 1 else beta
        if bias.shape == (batch, m, n):
            cb, cm, cn = bias.stride()
        else:
            cb, cm, cn = bias.expand(batch, m, n).stride()
    else:
        cb = cm = cn = 0
        bias = out
    if path == "tma":
        _launch_tma(A, B, out, bias, alpha, beta, (cb, cm, cn), has_bias)
    elif path == "gemv":
        if m < n:
            matrix, vector = B, A
            rows, count = n, m
            xm, xk, vr, vk = B.stride(2), B.stride(1), A.stride(1), A.stride(2)
            ym, yv, cm, cn = 1, n, cn, cm
        else:
            matrix, vector = A, B
            rows, count = m, n
            xm, xk, vr, vk = A.stride(1), A.stride(2), B.stride(2), B.stride(1)
            ym, yv = n, 1
        if m < n and xm == 1 and n >= 32 and 1024 <= k <= 4096:
            # Adjacent outputs share a cache line of row-major B.
            rows_per_cta = 8 if batch >= 8 else 4
        else:
            rows_per_cta = 4 if rows >= 512 else 1
        _batch_gemv[(triton.cdiv(rows, rows_per_cta), count, batch)](
            matrix,
            vector,
            bias,
            out,
            rows,
            k,
            xm,
            xk,
            matrix.stride(0),
            vr,
            vk,
            vector.stride(0),
            ym,
            yv,
            m * n,
            cm,
            cn,
            cb,
            alpha,
            beta,
            has_bias,
            has_bias and beta != 0,
            rows_per_cta,
            triton.next_power_of_2(k),
        )
    else:
        if k <= 128:
            bm, bn, bk = 16, 32, max(16, min(64, triton.next_power_of_2(k)))
        elif m % 64 == 0 and n % 64 == 0 and batch * (m // 64) * (n // 64) >= 128:
            # A complete wave of large output tiles pays for a wider MMA.
            bm, bn, bk = 64, 64, 64
        elif m <= 64 or n < 64:
            bm, bn, bk = 16, 16 if n % 16 else 32, 128
        else:
            bm, bn, bk = 64, 64, 64
        _batch_mm[(triton.cdiv(m, bm), triton.cdiv(n, bn), batch)](
            A,
            B,
            bias,
            out,
            m,
            n,
            k,
            *A.stride(),
            *B.stride(),
            cb,
            cm,
            cn,
            alpha,
            beta,
            has_bias,
            has_bias and beta != 0,
            bm,
            bn,
            bk,
            num_stages=3,
            num_warps=4,
        )
    return out


def bmm(A, B):
    path = _batch_path(A, B)
    if path is not None:
        return _batch_product(A, B, path)
    return generic_bmm(A, B)
