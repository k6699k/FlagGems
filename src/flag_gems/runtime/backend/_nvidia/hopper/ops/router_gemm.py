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

from flag_gems.runtime import torch_device_fn

from .mm import router_gemm as _fallback


@triton.jit
def _router_gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.program_id(1)
    k = tl.arange(0, triton.next_power_of_2(K))
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    w = tl.load(W + expert * K + k, k < K, 0).to(tl.float32)
    tl.store(Y + row * N + expert, tl.sum(x * w, axis=0))


@triton.jit
def _router_dot(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    k = split * BK + tl.arange(0, BK)
    x_ptr = X + row[:, None] * K + k[None, :]
    w_ptr = W + col[:, None] * K + k[None, :]
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in range(K // (BK * SPLITS)):
        x = tl.load(x_ptr, row[:, None] < M, 0)
        w = tl.load(w_ptr, col[:, None] < N, 0)
        acc = tl.dot(x, tl.trans(w), acc)
        x_ptr += BK * SPLITS
        w_ptr += BK * SPLITS
    offsets = (split * M + row[:, None]) * N + col[None, :]
    tl.store(Y + offsets, acc, (row[:, None] < M) & (col[None, :] < N))


@triton.jit
def _router_reduce(P, Y, SIZE: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, S)
    partial = tl.load(P + s[:, None] * SIZE + i[None, :], i[None, :] < SIZE, 0)
    tl.store(Y + i, tl.sum(partial, 0), i < SIZE)


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return FP32 router logits for a token matrix and expert weights [N, K]."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("router_gemm expects two matrices")
    m, k = x.shape
    n, weight_k = weight.shape
    if k != weight_k:
        raise ValueError("router_gemm requires matching input and weight K dimensions")
    if x.device != weight.device:
        raise ValueError("router_gemm requires inputs on the same device")
    if not (
        x.is_contiguous()
        and weight.is_contiguous()
        and x.dtype == weight.dtype
        and x.dtype in (torch.float16, torch.bfloat16)
        and 0 < n <= 512
        and 0 < m
        and 0 < k <= 8192
        and k % 128 == 0
        and max(m * k, m * n) < 2**31
    ):
        return _fallback(x, weight)
    y = torch.empty((m, n), dtype=torch.float32, device=x.device)
    with torch_device_fn.device(x.device):
        # One CTA per logit avoids padded tensor-core tiles for tiny gates.
        if m <= 16 or (n <= 8 and m <= 256):
            _router_gemv[(m, n)](x, weight, y, n, k, num_warps=4)
            return y

        bn = min(64, max(16, triton.next_power_of_2(n)))
        bm = 16 if n <= 16 and m >= 2048 else 64
        tiles_m, tiles_n = triton.cdiv(m, bm), triton.cdiv(n, bn)
        # Below one wave of output tiles, split K for concurrency and reduce
        # FP32 partials in a fixed order. Larger grids write logits directly.
        if tiles_m * tiles_n >= 128:
            bk = 256 if k % 256 == 0 else 128
            splits = 1
        else:
            bk = 128
            splits = 8
            while k % (bk * splits):
                splits //= 2
        partials = (
            torch.empty((splits, m, n), dtype=y.dtype, device=y.device)
            if splits > 1
            else y
        )
        _router_dot[(tiles_m, tiles_n, splits)](
            x,
            weight,
            partials,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            splits,
            num_warps=4,
        )
        if splits > 1:
            _router_reduce[(triton.cdiv(m * n, 256),)](partials, y, m * n, splits, 256)
    return y
