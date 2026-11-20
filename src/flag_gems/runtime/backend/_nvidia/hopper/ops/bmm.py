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

from flag_gems.ops.bmm import bmm as generic_bmm
from flag_gems.runtime import torch_device_fn


@triton.jit
def _store_result(
    value,
    bias,
    out,
    offsets,
    mask,
    bias_offsets,
    alpha,
    beta,
    HAS_BIAS: tl.constexpr,
    READ_BIAS: tl.constexpr,
):
    if HAS_BIAS:
        value *= alpha
        if READ_BIAS:
            value += beta * tl.load(bias + bias_offsets, mask, other=0).to(tl.float32)
    tl.store(out + offsets, value, mask)


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
    _store_result(
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


@triton.jit
def _batch_mm(
    a,
    b,
    bias,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sa: tl.constexpr,
    am: tl.constexpr,
    ak: tl.constexpr,
    sb: tl.constexpr,
    bk: tl.constexpr,
    bn: tl.constexpr,
    cb: tl.constexpr,
    cm: tl.constexpr,
    cn: tl.constexpr,
    alpha,
    beta,
    HAS_BIAS: tl.constexpr,
    READ_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    if M * am >= 2**31 or M * cm >= 2**31:
        m = m.to(tl.int64)
    if N * bn >= 2**31 or N * cn >= 2**31:
        n = n.to(tl.int64)
    if BK * ak >= 2**31 or BK * bk >= 2**31:
        k = k.to(tl.int64)
    ap = a + batch * sa + m[:, None] * am + k[None, :] * ak
    bp = b + batch * sb + k[:, None] * bk + n[None, :] * bn
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(tl.cdiv(K, BK)):
        x = tl.load(ap, (m[:, None] < M) & (i * BK + k[None, :] < K), other=0)
        y = tl.load(bp, (n[None, :] < N) & (i * BK + k[:, None] < K), other=0)
        acc = tl.dot(x, y, acc)
        ap += BK * ak
        bp += BK * bk
    mask = (m[:, None] < M) & (n[None, :] < N)
    offsets = batch * M * N + m[:, None] * N + n[None, :]
    bias_offsets = batch * cb + m[:, None] * cm + n[None, :] * cn
    _store_result(
        acc, bias, out, offsets, mask, bias_offsets, alpha, beta, HAS_BIAS, READ_BIAS
    )


def _supported(A, B):
    if A.ndim != 3 or B.ndim != 3:
        return False
    batch, m, k = A.shape
    b, other_k, n = B.shape
    if not (0 < batch == b <= 65535 and 0 < k == other_k and min(m, n) > 0):
        return False
    if A.dtype not in (torch.float16, torch.bfloat16) or A.dtype != B.dtype:
        return False
    if A.device != B.device or A.stride(2) != 1 or min(B.stride()[1:]) != 1:
        return False
    # The GEMV reduction keeps a complete row in registers. For a matrix
    # product, restrict small tiles to problems with few independent CTAs.
    return (min(m, n) <= 4 and 256 <= k <= 16384) or (
        m <= 512
        and n <= 512
        and (1024 <= k <= 8192 or (16 <= k <= 128 and min(m, n) >= 16))
    )


def _batch_product(A, B, bias=None, alpha=1, beta=0):
    batch, m, k = A.shape
    n = B.shape[2]
    out = torch.empty((batch, m, n), device=A.device, dtype=A.dtype)
    has_bias = bias is not None
    if has_bias:
        cb, cm, cn = bias.expand(batch, m, n).stride()
    else:
        cb = cm = cn = 0
        bias = out
    with torch_device_fn.device(A.device):
        if min(m, n) <= 4:
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
                bm, bn, bk = 16, 32, min(64, triton.next_power_of_2(k))
            elif m % 64 == 0 and n % 64 == 0 and batch * (m // 64) * (n // 64) >= 128:
                # A complete wave of large output tiles pays for a wider MMA.
                bm, bn, bk = 64, 64, 64
            else:
                bm, bn, bk = 16, 16 if n % 16 else 32, 128
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
    if _supported(A, B):
        return _batch_product(A, B)
    return generic_bmm(A, B)
