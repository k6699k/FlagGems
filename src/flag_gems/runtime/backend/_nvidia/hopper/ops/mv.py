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

from flag_gems.ops.mv import mv as generic_mv
from flag_gems.runtime import torch_device_fn


@triton.jit
def _row_mv(A, X, Y, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BK)
    a = tl.load(A + row * K + col, col < K, 0)
    x = tl.load(X + col, col < K, 0)
    tl.store(Y + row, tl.sum(a.to(tl.float32) * x.to(tl.float32), 0))


@triton.jit
def _column_mv(
    A,
    X,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    ks = split * CHUNK + tl.arange(0, BK)
    acc = tl.full((BK, BN), 0, tl.float32)
    for i in range(triton.cdiv(CHUNK, BK)):
        col = ks + i * BK
        a = tl.load(
            A + col[:, None] * N + row[None, :],
            (row[None, :] < N) & (col[:, None] < K),
            0,
        )
        x = tl.load(X + col, col < K, 0)
        acc += a.to(tl.float32) * x.to(tl.float32)[:, None]
    tl.store(Y + split * N + row, tl.sum(acc, 0), row < N)


@triton.jit
def _finish_mv(
    P, Y, N: tl.constexpr, S: tl.constexpr, BN: tl.constexpr, BS: tl.constexpr
):
    r = tl.program_id(0) * BN + tl.arange(0, BN)
    s = tl.arange(0, BS)
    v = tl.load(P + s[:, None] * N + r[None, :], (s[:, None] < S) & (r[None, :] < N), 0)
    tl.store(Y + r, tl.sum(v, 0), r < N)


def mv(inp, vec):
    if inp.ndim != 2 or vec.ndim != 1:
        raise RuntimeError("mv expects a matrix and a vector")
    rows, k = inp.shape
    if k != vec.numel() or inp.dtype != vec.dtype or inp.device != vec.device:
        raise RuntimeError("mv needs matching K dimensions, dtypes and devices")
    if (
        inp.device.type != "cuda"
        or inp.dtype not in (torch.float16, torch.bfloat16)
        or not vec.is_contiguous()
        or not (1024 <= k <= 8192 and 0 < rows and rows * k < 2**31)
    ):
        return generic_mv(inp, vec)

    row_major = inp.is_contiguous()
    if not row_major and (inp.stride() != (1, rows) or (rows < 256 and k < 3072)):
        return generic_mv(inp, vec)

    if inp.device.index != torch_device_fn.current_device():
        with torch_device_fn.device(inp.device):
            return mv(inp, vec)
    out = torch.empty(rows, device=inp.device, dtype=inp.dtype)
    if row_major:
        # A CTA can reduce one contiguous row without exchanging partials.
        _row_mv[(rows,)](inp, vec, out, k, triton.next_power_of_2(k), num_warps=8)
    elif (
        rows >= 256
        and rows % 64 == 0
        and k % 256 == 0
        and (rows < 2048 or k <= 2 * rows)
    ):
        # Reading several adjacent rows together coalesces the column-major
        # loads. Reduce all of K in this CTA and write directly to out.
        if rows >= 16384:
            tile_rows, tile_k, warps = 64, 256, 4
        elif rows >= 8192:
            tile_rows, tile_k, warps = 64, 512, 8
        elif rows >= 2048:
            tile_rows = 16 if rows == 2048 and k <= 2048 else 32
            tile_k, warps = (1024 if k >= 4096 else 512), 8
        else:
            tile_rows, tile_k, warps = 8, min(2048, triton.next_power_of_2(k)), 8
        _column_mv[(rows // tile_rows, 1)](
            inp,
            vec,
            out,
            rows,
            k,
            tile_rows,
            tile_k,
            k,
            num_warps=warps,
        )
    else:
        # Load neighboring output rows together. Splitting K provides enough
        # CTAs even for a narrow matrix and reduces long-stride HBM reads.
        tile_rows = 16 if rows < 256 else 64
        chunk = 128 if rows < 4096 else 512
        splits = triton.cdiv(k, chunk)
        partials = torch.empty((splits, rows), device=inp.device, dtype=torch.float32)
        _column_mv[(triton.cdiv(rows, tile_rows), splits)](
            inp,
            vec,
            partials,
            rows,
            k,
            tile_rows,
            64,
            chunk,
        )
        _finish_mv[(triton.cdiv(rows, 32),)](
            partials,
            out,
            rows,
            splits,
            32,
            triton.next_power_of_2(splits),
        )
    return out
