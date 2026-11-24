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

import logging
from pathlib import Path

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.flagtune import register_flagtune_op
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import LibTuner

from .bmm import _FP32_BYTES, _UB_BYTES, is_ascend_910b, vector_tile_fits_910b

logger = logging.getLogger(__name__)
_MV_EXPAND_PATH = str(Path(__file__).resolve().parent.parent / "mv_ascend_expand.yaml")
_MV_MAX_GRID = 65535
# BLOCK_M=128, BLOCK_N=16 fits 910B UB even with a strided reduction.
_MV_910B_MAX_ROWS = 16 * _MV_MAX_GRID
register_flagtune_op(
    "mv_column", description="matrix-vector multiplication over a column-major matrix"
)


def _prune_mv_configs(configs, named_args, **kwargs):
    """Bound the legacy GEMV launch grid and FP32 accumulation tile."""
    args = {**named_args, **kwargs}
    strided_reduction = args["stride_am"] > 1
    dtype = args["A"].dtype
    is_910b = is_ascend_910b()
    low_precision = dtype in (torch.float16, torch.bfloat16)
    # The legacy kernel accumulates a full FP32 tile. Strided loads
    # along K leave less UB available than contiguous reductions.
    tile_limit = 4096 if strided_reduction else 32768
    valid = []
    for config in configs:
        block_m, block_n = (config.kwargs[key] for key in ("BLOCK_M", "BLOCK_N"))
        if triton.cdiv(args["N"], block_n) > _MV_MAX_GRID:
            continue
        if is_910b:
            # The legacy strided gather needs additional UB scratch.
            if (
                strided_reduction
                and max(block_n, 2) * block_m * _FP32_BYTES >= _UB_BYTES // 12
            ):
                continue
            if vector_tile_fits_910b(
                block_n,
                block_m,
                dtype.itemsize,
                config.num_stages,
            ):
                valid.append(config)
            continue
        rows = max(block_n, 2) if strided_reduction else block_n
        if not low_precision or block_m * rows < tile_limit:
            valid.append(config)
    return valid


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mv"),
    key=["M", "N", "stride_an", "stride_am", "stride_bm", "stride_cn"],
    prune_configs_by={"early_config_prune": _prune_mv_configs},
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    flagtune_op_name="mv",
    flagtune_expand_op_name="mv_ascend",
    flagtune_yaml_path=_MV_EXPAND_PATH,
    flagtune_op_id="flaggems/mv",
    flagtune_variant="gemv",
    policy="flagtune",
)
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = ext.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)


def _launch_general_mv(inp, vec, out):
    n, k = inp.shape
    max_rows = _MV_910B_MAX_ROWS if is_ascend_910b() else n
    for start in range(0, n, max_rows):
        rows = min(n - start, max_rows)
        matrix, result = inp, out
        if rows < n:
            matrix = inp[start : start + rows]
            result = out[start : start + rows]

        grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_N"]),)
        mv_kernel[grid](
            matrix,
            vec,
            result,
            rows,
            k,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
        )


@triton.jit
def _mv_column_tiles(
    A,
    X,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_K: tl.constexpr,
    STRIDE_X: tl.constexpr,
    PADDED_N: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_tiles = tl.cdiv(N, BLOCK_N)
    rows = (pid % row_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
    part = pid // row_tiles
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K, BLOCK_N), tl.float32)
    chunk = CHUNK_K if SPLIT else K
    # Adjacent output rows are adjacent in memory for column-major inputs.
    for tile in range(tl.cdiv(chunk, BLOCK_K)):
        ks = part * chunk + tile * BLOCK_K + offsets
        a = tl.load(
            A + ks[:, None] * STRIDE_K + rows[None, :],
            (ks < K)[:, None] & (rows < N)[None, :],
            other=0,
        ).to(tl.float32)
        x = tl.load(X + ks * STRIDE_X, ks < K, other=0).to(tl.float32)
        acc += a * x[:, None]
    values = tl.sum(acc, axis=0)
    if SPLIT:
        tl.store(Y + part * PADDED_N + rows, values)
    else:
        tl.store(Y + rows, values, rows < N)


@triton.jit
def _mv_merge_column_tiles(
    Partial,
    Out,
    N: tl.constexpr,
    PADDED_N: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    parts = tl.arange(0, BLOCK_SPLITS)
    partial = tl.load(
        Partial + parts[:, None] * PADDED_N + rows[None, :],
        (parts < SPLITS)[:, None] & (rows < N)[None, :],
        other=0,
    )
    tl.store(Out + rows, tl.sum(partial, axis=0), rows < N)


_COLUMN_MIN_CHUNK = 128
_COLUMN_MAX_BLOCK_N = 256
_COLUMN_SPLIT_ROWS = 4096
_COLUMN_SPLIT_BLOCK_N = 64
_COLUMN_DIRECT_BLOCK_N = 128
_COLUMN_DEFAULT_BLOCK_K = 128


def _prune_column_mv_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    n, k, split = (args[name] for name in ("N", "K", "SPLIT"))
    is_910b = is_ascend_910b()
    candidates = []
    for config in configs:
        block_n, block_k, chunk = (
            config.kwargs[name] for name in ("BLOCK_N", "BLOCK_K", "CHUNK_K")
        )
        if block_n > _COLUMN_MAX_BLOCK_N or bool(chunk) != bool(split):
            continue
        if split and (chunk < _COLUMN_MIN_CHUNK or chunk % block_k):
            continue
        if is_910b:
            if not vector_tile_fits_910b(
                block_n, block_k, args["A"].dtype.itemsize, config.num_stages
            ):
                continue
        elif block_n * block_k > (8192 if config.num_stages > 1 else 16384):
            continue
        splits = triton.cdiv(k, chunk) if split else 1
        if triton.cdiv(n, block_n) * splits <= _MV_MAX_GRID:
            candidates.append(config)
    return candidates


mv_column_kernel = libentry()(
    libtuner(
        configs=[
            triton.Config(
                {
                    "BLOCK_N": block_n,
                    "BLOCK_K": _COLUMN_DEFAULT_BLOCK_K,
                    "CHUNK_K": chunk,
                },
                num_warps=4,
                num_stages=1,
            )
            for block_n, chunk in (
                (_COLUMN_SPLIT_BLOCK_N, _COLUMN_MIN_CHUNK),
                (_COLUMN_DIRECT_BLOCK_N, 0),
            )
        ],
        key=["N", "K", "STRIDE_K", "STRIDE_X", "SPLIT"],
        prune_configs_by={"early_config_prune": _prune_column_mv_configs},
        warmup=5,
        rep=20,
        use_cuda_graph=True,
        disable_aabs=True,
        flagtune_op_name="mv_column",
        flagtune_expand_op_name="mv_column_ascend",
        flagtune_yaml_path=_MV_EXPAND_PATH,
        flagtune_op_id="flaggems/mv",
        flagtune_variant="column_gemv",
        flagtune_dtype_resolver=lambda args: (
            args["A"].dtype,
            args["X"].dtype,
            args["A"].dtype,
        ),
        policy="flagtune",
    )(_mv_column_tiles)
)


def _can_launch_column_mv(inp, vec):
    n, k = inp.shape
    splits = triton.cdiv(k, _COLUMN_MIN_CHUNK) if n < _COLUMN_SPLIT_ROWS else 1
    block_n = _COLUMN_SPLIT_BLOCK_N if splits > 1 else _COLUMN_DIRECT_BLOCK_N
    return (
        n > 1
        and k >= 1024
        and inp.stride(0) == 1
        and inp.stride(1) == n
        and inp.dtype == vec.dtype
        and inp.dtype in (torch.float16, torch.bfloat16)
        and splits <= 128
        and triton.cdiv(n, block_n) * splits <= _MV_MAX_GRID
        and n * k <= 2**31
        and (k - 1) * vec.stride(0) < 2**31
    )


def _launch_column_mv(inp, vec, out, n, k):
    split = int(n < _COLUMN_SPLIT_ROWS)
    chunk = _COLUMN_MIN_CHUNK if split else 0
    block_n = _COLUMN_SPLIT_BLOCK_N if split else _COLUMN_DIRECT_BLOCK_N
    splits = triton.cdiv(k, chunk) if split else 1
    tune = LibTuner.should_tune(mv_column_kernel)
    # Reserve a row pitch large enough for every candidate's padded output tile.
    row_alignment = _COLUMN_MAX_BLOCK_N if tune else block_n
    padded_n = triton.cdiv(n, row_alignment) * row_alignment
    partial = out
    if split:
        partial = torch.empty(
            (splits, padded_n), device=out.device, dtype=torch.float32
        )
    args = (
        inp,
        vec,
        partial,
        n,
        k,
        inp.stride(1),
        vec.stride(0),
        padded_n,
        split,
    )
    if tune:

        def grid(meta):
            parts = triton.cdiv(k, meta["CHUNK_K"]) if split else 1
            return (triton.cdiv(n, meta["BLOCK_N"]) * parts,)

        _, selected = mv_column_kernel[grid](*args)
        if split:
            splits = triton.cdiv(k, selected["CHUNK_K"])
    else:
        _mv_column_tiles[(triton.cdiv(n, block_n) * splits,)](
            *args, block_n, _COLUMN_DEFAULT_BLOCK_K, chunk, num_warps=4, num_stages=1
        )
    if split:
        _mv_merge_column_tiles[(triton.cdiv(n, 64),)](
            partial,
            out,
            n,
            partial.shape[1],
            splits,
            triton.next_power_of_2(splits),
            64,
            num_warps=4,
            num_stages=1,
        )


def mv(inp, vec):
    logger.debug("GEMS_ASCEND MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    if M == 0:
        return torch.zeros((N,), device=inp.device, dtype=inp.dtype)
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    if N == 0:
        return out
    with torch_device_fn.device(inp.device):
        if _can_launch_column_mv(inp, vec):
            _launch_column_mv(inp, vec, out, N, M)
        else:
            _launch_general_mv(inp, vec, out)
    return out
