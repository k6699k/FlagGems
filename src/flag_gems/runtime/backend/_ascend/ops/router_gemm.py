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

from pathlib import Path

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.flagtune import register_flagtune_op
from flag_gems.utils import libentry, libtuner

from .bmm import AscendBlasTuner, cube_tile_fits_910b, tune_fixed_gemm

EXPAND_CONFIG_FILENAME = str(
    Path(__file__).resolve().parent.parent / "router_gemm_ascend_expand.yaml"
)
register_flagtune_op("router_gemm", default=False, description="router matrix multiply")


@libentry()
@triton.jit
def _router_vector(
    X,
    W,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    XM: tl.constexpr,
    XK: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    BK: tl.constexpr,
    SIZE: tl.constexpr,
    GRID: tl.constexpr,
):
    first = tl.program_id(0)
    k = tl.arange(0, BK)
    end = SIZE if GRID < SIZE else first + 1
    for pid in range(first, end, GRID):
        m = pid // N
        n = pid % N
        xp = X + m * XM + k * XK
        wp = W + n * WN + k * WK
        acc = tl.zeros((BK,), tl.float32)
        for start in range(0, K, BK):
            valid = start + k < K
            x = tl.load(xp, valid, 0).to(tl.float32)
            w = tl.load(wp, valid, 0).to(tl.float32)
            acc += x * w
            xp += BK * XK
            wp += BK * WK
        tl.store(Y + pid, tl.sum(acc, 0))


@libentry()
@triton.jit
def _router_cube(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    XM: tl.constexpr,
    XK: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)
    z = tl.program_id(1)
    m = (pid // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    n = (pid % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    steps = tl.cdiv(K, BK * SPLIT)
    k = z * steps * BK + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(steps):
        a = tl.load(
            X + m[:, None] * XM + k[None, :] * XK,
            (m[:, None] < M) & (k[None, :] < K),
            0,
        )
        if WN == 1:
            b = tl.load(
                W + k[:, None] * WK + n[None, :],
                (k[:, None] < K) & (n[None, :] < N),
                0,
            )
        else:
            w = tl.load(
                W + n[:, None] * WN + k[None, :] * WK,
                (n[:, None] < N) & (k[None, :] < K),
                0,
            )
            b = tl.trans(w)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        k += BK
    tl.store(
        Y + z * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@libentry()
@triton.jit
def _router_reduce(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for z in range(SPLIT):
        value += tl.load(P + z * SIZE + i, i < SIZE, 0)
    tl.store(Y + i, value, i < SIZE)


def _router_plan(x, weight):
    M, K = x.shape
    N = weight.shape[0]
    weight_strides = weight.stride()
    dense_nn = N % 16 == 0 and weight_strides == (1, N)
    low_precision = x.dtype in (torch.float16, torch.bfloat16)
    small_short_k_nt = (
        low_precision
        and not dense_nn
        and M <= 16
        and K == 2048
        and N >= 256
        and N % 32 == 0
    )
    medium_short_k = low_precision and 512 <= M <= 1024 and K == 2048 and N % 256 == 0
    wide_bf16_nt = x.dtype == torch.bfloat16 and N >= 256 and not dense_nn
    # Cube reads either dense weight layout directly.
    vector_fallback = (
        K == 0
        or K % 128 != 0
        or (N != 8 and N % 16 != 0)
        or not x.is_contiguous()
        or not (weight.is_contiguous() or dense_nn)
        or x.data_ptr() % 16 != 0
        or weight.data_ptr() % 16 != 0
    )
    small_nt = M == 1 and not (dense_nn or wide_bf16_nt or small_short_k_nt)
    if vector_fallback or ((small_nt or (M <= 16 and N <= 8)) and K <= 8192):
        block_k = max(1, triton.next_power_of_2(K))
        return None, min(1024, block_k) if vector_fallback else block_k

    if small_short_k_nt:
        # Narrow N tiles expose parallel work without splitting K.
        bm, bn, bk = 16, 32, 512
    elif medium_short_k:
        bm, bn, bk = (64, 128, 256) if dense_nn else (128, 64, 512)
    elif low_precision and M >= 8192 and N <= 384 and N % 64 == 0 and K % 256 == 0:
        bm, bn, bk = 256, min(N, 128), 256
        if dense_nn:
            if K == 2048 and N % 256 == 0:
                bm, bn = 128, 256
            elif N < 256:
                bm = 128
    elif wide_bf16_nt:
        bm, bn, bk = min(128, max(16, triton.next_power_of_2(M))), 128, 512
    elif dense_nn:
        bm = 16 if M <= 16 else 32 if M < 128 else 128
        bn, bk = min(triton.next_power_of_2(N), 128), 256
    else:
        bm = 16 if M <= 32 else 32 if M < 128 else 64
        bn, bk = (16 if N <= 16 else 64), 128

    tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    # Target 20 output tasks on 910B, with at most eight K partitions.
    split = (
        min(8, triton.next_power_of_2(triton.cdiv(20, tiles)))
        if K >= 2048 and not (small_short_k_nt or medium_short_k)
        else 1
    )
    # Avoid masked or empty K partitions in the matrix pipeline.
    while bk > 128 and K % (bk * split) != 0:
        bk //= 2
    while split > 1 and K % (bk * split) != 0:
        split //= 2
    return (bm, bn, bk, split), None


def _can_tune_router_cube(m, n, split):
    # The smallest tile is a legal fallback for every shape in the search.
    return triton.cdiv(m, 16) * triton.cdiv(n, 16) * split <= 65535


def _prune_router_cube_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k, split = (args[name] for name in ("M", "N", "K", "SPLIT"))
    max_m = max(16, triton.next_power_of_2(m))
    max_n = max(16, triton.next_power_of_2(n))
    element_bytes = args["X"].dtype.itemsize
    valid = []
    for config in configs:
        bm, bn, bk = (config.kwargs[name] for name in ("BM", "BN", "BK"))
        if bm > max_m or bn > max_n or k % (bk * split):
            continue
        if cube_tile_fits_910b(bm, bn, bk, element_bytes, config.num_stages):
            valid.append(config)
    return valid


router_cube_tuned = libentry()(
    libtuner(
        configs=[
            triton.Config(dict(BM=bm, BN=bn, BK=bk), num_warps=32, num_stages=2)
            for bm, bn in (
                (16, 16),
                (16, 32),
                (16, 64),
                (16, 128),
                (32, 16),
                (32, 64),
                (32, 128),
                (64, 16),
                (64, 64),
                (64, 128),
                (128, 16),
                (128, 32),
                (128, 64),
                (128, 128),
                (256, 64),
                (256, 128),
            )
            for bk in (128, 256, 512)
        ],
        key=["M", "N", "K", "XM", "XK", "WN", "WK", "SPLIT"],
        prune_configs_by={"early_config_prune": _prune_router_cube_configs},
        use_cuda_graph=True,
        disable_aabs=True,
        policy=AscendBlasTuner,
        flagtune_op_name="router_gemm",
        flagtune_op_id="flaggems/router_gemm",
        flagtune_variant="cube",
        flagtune_expand_op_name="router_gemm_ascend",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
        flagtune_dtype_resolver=lambda args: tuple(
            args[k].dtype for k in ("X", "W", "Y")
        ),
    )(_router_cube.jit_function)
)


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute router scores with FP32 accumulation and output."""
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("router_gemm expects x[M, K] and weight[N, K]")
    if x.device != weight.device or x.dtype != weight.dtype:
        raise ValueError("router_gemm inputs must have the same device and dtype")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("router_gemm supports float16, bfloat16 and float32")
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if m == 0 or n == 0:
        return out
    size = m * n
    strides = (*x.stride(), *weight.stride())
    with torch_device_fn.device(x.device):
        tile, vector_block_k = _router_plan(x, weight)
        if tile is None:
            # Ascend launch dimensions must fit in 16 bits.
            grid = size if size <= 65535 else 4096
            _router_vector[(grid,)](
                x, weight, out, n, k, *strides, vector_block_k, size, grid
            )
            return out

        bm, bn, bk, split = tile
        partial = (
            torch.empty((split, m, n), device=x.device, dtype=torch.float32)
            if split > 1
            else out
        )
        args = (x, weight, partial, m, n, k, *strides)
        if _can_tune_router_cube(m, n, split) and tune_fixed_gemm(router_cube_tuned):

            def grid(meta):
                return (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]), split)

            router_cube_tuned[grid](*args, SPLIT=split)
        else:
            tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
            _router_cube[(tiles, split)](*args, bm, bn, bk, split)
        if split > 1:
            _router_reduce[(triton.cdiv(size, 1024),)](partial, out, size, split, 1024)
    return out
