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
import triton.language.extra.cann.extension as al

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.runtime.backend._ascend.blas_resources import (
    cube_tile_fits_910b,
    is_ascend_910b,
    prune_cube_tiles_910b,
    vector_tile_fits_910b,
)
from flag_gems.runtime.backend._ascend.gemm_tuning import (
    AscendBlasTuner,
    _blockified,
    launch_panel_gemm,
    make_fixed_gemm_tuner,
    prune_panel_gemm,
    tune_fixed_gemm,
)
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.libentry import LibTunerRunMode

from .bmm import _launch_tiled_gemm, _make_tiled_tuner, _select_direct_nt_tile
from .copy import copy as copy_tensor

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = str(
    Path(__file__).resolve().parent.parent / "mm_ascend_expand.yaml"
)


def _prune_mm_configs(configs, named_args, **kwargs):
    if not configs:
        return configs
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    if 256 <= m <= 512 and 256 <= n <= 512 and k >= 128:
        preferred = [
            config
            for config in configs
            if (
                config.kwargs["BLOCK_M"],
                config.kwargs["BLOCK_N"],
                config.kwargs["BLOCK_K"],
            )
            == (64, 128, 128)
        ]
        if preferred:
            return preferred
    large_aligned = min(m, n) >= 1024 and m % 128 == 0 and n % 128 == 0
    if not large_aligned:
        compact = [
            config
            for config in configs
            if config.kwargs["BLOCK_M"] <= 128
            and config.kwargs["BLOCK_N"] <= 128
            and config.kwargs["BLOCK_K"] <= 256
        ]
        configs = compact or configs
    program_counts = {
        id(config): triton.cdiv(m, config.kwargs["BLOCK_M"])
        * triton.cdiv(n, config.kwargs["BLOCK_N"])
        for config in configs
    }
    min_programs = min(program_counts.values())
    max_programs = max(1, 2 * min_programs)
    candidates = [
        config for config in configs if program_counts[id(config)] <= max_programs
    ]
    if large_aligned:
        if k >= 1024:
            if args["A"].dtype == torch.float32:
                preferred = [
                    config
                    for config in candidates
                    if (
                        config.kwargs["BLOCK_M"],
                        config.kwargs["BLOCK_N"],
                        config.kwargs["BLOCK_K"],
                    )
                    in ((128, 256, 128), (256, 128, 128))
                ]
                if preferred:
                    return preferred
                preferred_block_k = (256,)
            elif args["A"].dtype in (torch.float16, torch.bfloat16):
                preferred = [
                    config
                    for config in candidates
                    if (
                        config.kwargs["BLOCK_M"],
                        config.kwargs["BLOCK_N"],
                        config.kwargs["BLOCK_K"],
                    )
                    in ((128, 256, 256), (256, 128, 256))
                ]
                if preferred:
                    return preferred
                preferred_block_k = (256,)
            else:
                preferred_block_k = (256,)
        else:
            preferred_block_k = (128,)
        preferred = [
            config
            for config in candidates
            if config.kwargs["BLOCK_K"] in preferred_block_k
        ]
        if preferred:
            return preferred
    return candidates


class _MMFlagTuneTuner(AscendBlasTuner):
    def __init__(self, *args, **kwargs):
        kwargs["prune_configs_by"] = {"early_config_prune": self._prune_candidates}
        super().__init__(*args, **kwargs)

    def _prune_candidates(self, configs, named_args, **kwargs):
        args = {**named_args, **kwargs}
        if is_ascend_910b():
            # The kernel iterates over logical tiles with a capped launch grid.
            valid = prune_cube_tiles_910b(
                (c for c in configs if c.kwargs.get("SPLIT_K", 1) == 1),
                args["A"].dtype,
                ("BLOCK_M", "BLOCK_N", "BLOCK_K"),
            )
            if (
                self._flagtune_mode is runtime.TuningMode.DEFAULT
                and self._run_mode is not LibTunerRunMode.EXHAUSTIVE_COLLECTION
            ):
                return _prune_mm_configs(valid, named_args, **kwargs)
            return valid
        if (
            self._flagtune_mode is runtime.TuningMode.DEFAULT
            and getattr(self, "_run_mode", None)
            is not LibTunerRunMode.EXHAUSTIVE_COLLECTION
        ):
            return _prune_mm_configs(configs, named_args, **kwargs)

        m, n = args["M"], args["N"]
        target = triton.runtime.driver.active.get_current_target()
        is_910c = str(target.arch).startswith("Ascend910_93")
        valid = []
        for config in configs:
            bm = config.kwargs["BLOCK_M"]
            bn = config.kwargs["BLOCK_N"]
            # The public kernel writes directly to the output. Split-K needs
            # a separately initialized accumulator before it can be enabled.
            if config.kwargs.get("SPLIT_K", 1) != 1:
                continue
            if triton.cdiv(m, bm) * triton.cdiv(n, bn) > 65535:
                continue
            # A single Cube tile holds a full FP32 accumulator. On 910C
            # its 128 KiB C buffer can hold at most 32768 FP32 values.
            if is_910c and bm * bn > 32768:
                continue
            valid.append(config)
        return valid


def _mm_heur_divisible_m(args):
    return args["M"] % args["BLOCK_M"] == 0


def _mm_heur_divisible_n(args):
    return args["N"] % args["BLOCK_N"] == 0


def _mm_capped_grid(num_tiles):
    return (min(num_tiles, _MAX_CORE_DIM),)


def _mm_grid(M, N, meta):
    num_tiles = triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"])
    return _mm_capped_grid(num_tiles * meta.get("SPLIT_K", 1))


# Ascend's launch coreDim is uint16. Keep the grid bounded even when the
# compiler cannot automatically map logical programs onto physical cores.
_MAX_CORE_DIM = 65535


_MM_HEURISTICS = {
    **_hcu.HEURISTICS_CONFIGS["mm"],
    "DIVISIBLE_M": _mm_heur_divisible_m,
    "DIVISIBLE_N": _mm_heur_divisible_n,
}


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    use_cuda_graph=True,
    policy=_MMFlagTuneTuner,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_ascend",
    flagtune_op_id="flaggems/mm",
    flagtune_variant="ascend_gemm",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.heuristics(_MM_HEURISTICS)
@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    # Preserve the base pointers before the per-tile computation advances them.
    A_base = A
    B_base = B
    C_base = C
    # Flatten both the output-tile and split-K dimensions. Capping only one
    # axis of a 2D grid would still allow their product to exceed coreDim.
    for task_id in range(
        tl.program_id(0), grid_m * grid_n * SPLIT_K, tl.num_programs(0)
    ):
        pid = task_id // SPLIT_K
        pid_z = task_id % SPLIT_K
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)
        ram = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rbn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            if EVEN_K:
                a = tl.load(A, mask=(ram < M)[:, None], other=0.0)
                b = tl.load(B, mask=(rbn < N)[None, :], other=0.0)
            else:
                k_remaining = K - k * (BLOCK_K * SPLIT_K)
                a = tl.load(
                    A,
                    mask=(rk[None, :] < k_remaining) & (ram < M)[:, None],
                    other=0.0,
                )
                b = tl.load(
                    B,
                    mask=(rk[:, None] < k_remaining) & (rbn < N)[None, :],
                    other=0.0,
                )
            if a.dtype != b.dtype:
                a = a.to(C.dtype.element_ty)
                b = b.to(C.dtype.element_ty)
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
        acc = acc.to(C.dtype.element_ty)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        if SPLIT_K == 1:
            tl.store(C, acc, mask=mask)
        else:
            tl.atomic_add(C, acc, mask=mask)
        # Restore scalar pointers so the next tile starts from the same bases.
        A = A_base
        B = B_base
        C = C_base


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


@libentry()
@triton.jit
def _mm_narrow_n_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # A few output columns do not fill a Cube tile. Reduce independent
    # output elements on Vector cores, accumulating products in FP32.
    for pid in range(tl.program_id(0), tl.cdiv(M, BLOCK_OUT), tl.num_programs(0)):
        rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        rk = tl.arange(0, BLOCK_K)
        row_acc = tl.zeros((BLOCK_OUT, BLOCK_K), tl.float32)
        for start in range(tl.cdiv(K, BLOCK_K)):
            kk = start * BLOCK_K + rk
            a_rows = tl.load(
                A + rows[:, None] * K + kk[None, :],
                mask=(rows < M)[:, None] & (kk < K)[None, :],
                other=0.0,
            ).to(tl.float32)
            b_vec = tl.load(B + kk, mask=kk < K, other=0.0).to(tl.float32)
            row_acc += a_rows * b_vec[None, :]
        tl.store(C + rows, tl.sum(row_acc, axis=1), mask=rows < M)


@libentry()
@triton.jit
def _mm_nt_split_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLITS: tl.constexpr,
):
    split = tl.program_id(1)
    for pid in range(
        tl.program_id(0), tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N), tl.num_programs(0)
    ):
        rows = pid // tl.cdiv(N, BLOCK_N) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid % tl.cdiv(N, BLOCK_N) * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in tl.range(tl.cdiv(K, BLOCK_K * SPLITS)):
            kk = (split * tl.cdiv(K, BLOCK_K * SPLITS) + start) * BLOCK_K + rk
            a = tl.load(
                A + rows[:, None] * K + kk[None, :],
                mask=(rows < M)[:, None] & (kk < K)[None, :],
                other=0.0,
            )
            bt = tl.load(
                B + cols[:, None] * K + kk[None, :],
                mask=(cols < N)[:, None] & (kk < K)[None, :],
                other=0.0,
            )
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(bt, "dot_pad_only_k")
            acc = tl.dot(a, tl.trans(bt), acc, allow_tf32=False)
        tl.store(
            C + split * M * N + rows[:, None] * N + cols[None, :],
            acc,
            mask=(rows < M)[:, None] & (cols < N)[None, :],
        )


@libentry()
@triton.jit
def _mm_nt_merge(P, C, MN: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    for pid in range(tl.program_id(0), tl.cdiv(MN, BLOCK), tl.num_programs(0)):
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        ss = tl.arange(0, SPLITS)
        partial = tl.load(P + ss[:, None] * MN + idx[None, :], idx[None, :] < MN, 0)
        out = tl.sum(partial, axis=0)
        tl.store(C + idx, out.to(C.dtype.element_ty), idx < MN)


@libentry()
@triton.jit
def _mm_nt_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for pid in range(
        tl.program_id(0), tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N), tl.num_programs(0)
    ):
        rows = pid // tl.cdiv(N, BLOCK_N) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid % tl.cdiv(N, BLOCK_N) * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in tl.range(tl.cdiv(K, BLOCK_K)):
            kk = start * BLOCK_K + rk
            a = tl.load(
                A + rows[:, None] * K + kk[None, :],
                mask=(rows < M)[:, None] & (kk < K)[None, :],
                other=0.0,
            )
            bt = tl.load(
                B + cols[:, None] * K + kk[None, :],
                mask=(cols < N)[:, None] & (kk < K)[None, :],
                other=0.0,
            )
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(bt, "dot_pad_only_k")
            acc = tl.dot(a, tl.trans(bt), acc, allow_tf32=False)
        tl.store(
            C + rows[:, None] * N + cols[None, :],
            acc.to(C.dtype.element_ty),
            mask=(rows < M)[:, None] & (cols < N)[None, :],
        )


@libentry()
@triton.jit
def _mm_rectangular_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for pid in range(
        tl.program_id(0), tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N), tl.num_programs(0)
    ):
        rows = pid // tl.cdiv(N, BLOCK_N) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid % tl.cdiv(N, BLOCK_N) * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in tl.range(tl.cdiv(K, BLOCK_K)):
            kk = start * BLOCK_K + rk
            a = tl.load(
                A + rows[:, None] * K + kk[None, :],
                mask=(rows < M)[:, None] & (kk < K)[None, :],
                other=0.0,
            )
            b = tl.load(
                B + kk[:, None] * N + cols[None, :],
                mask=(kk < K)[:, None] & (cols < N)[None, :],
                other=0.0,
            )
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(b, "dot_pad_only_k")
            acc = tl.dot(a, b, acc, allow_tf32=False)
        tl.store(
            C + rows[:, None] * N + cols[None, :],
            acc.to(C.dtype.element_ty),
            mask=(rows < M)[:, None] & (cols < N)[None, :],
        )


@triton.jit
def _mm_pack_narrow4_split_kernel(B, P, K: tl.constexpr, BK: tl.constexpr):
    for pid in range(tl.program_id(0), tl.cdiv(K, BK), tl.num_programs(0)):
        base = pid * BK
        idx = tl.arange(0, BK * 4)
        b = tl.load(B + base * 4 + idx).reshape((BK, 2, 2))
        b02, b13 = tl.split(b)
        b0, b2 = tl.split(b02)
        b1, b3 = tl.split(b13)
        k = base + tl.arange(0, BK)
        tl.store(P + k, b0)
        tl.store(P + K + k, b1)
        tl.store(P + 2 * K + k, b2)
        tl.store(P + 3 * K + k, b3)


@triton.jit
def _mm_pack_narrow_b_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr
):
    for pid in range(tl.program_id(0), tl.cdiv(K, BK), tl.num_programs(0)):
        k = pid * BK + tl.arange(0, BK)
        n = tl.arange(0, BN)
        b = tl.load(
            B + k[:, None] * N + n[None, :],
            (k < K)[:, None] & (n < N)[None, :],
            other=0,
        )
        tl.store(
            P + n[:, None] * K + k[None, :],
            tl.trans(b),
            (n < N)[:, None] & (k < K)[None, :],
        )


@triton.jit
def _mm_pack_narrow4_b_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr
):
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    n = tl.arange(0, BN)
    b = tl.load(
        B + k[:, None] * N + n[None, :], (k < K)[:, None] & (n < N)[None, :], other=0
    )
    # P is padded to 16 columns so the subsequent Cube load stays aligned.
    tl.store(P + n[:, None] * K + k[None, :], tl.trans(b), (k < K)[None, :])


@triton.jit
def _mm_packed_narrow_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    for pid in range(tl.program_id(0), tl.cdiv(M, BM), tl.num_programs(0)):
        rows = pid * BM + tl.arange(0, BM)
        cols = tl.arange(0, 16)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, 16), tl.float32)
        for s in tl.range(tl.cdiv(K, BK)):
            k = s * BK + rk
            a = tl.load(
                A + rows[:, None] * K + k[None, :],
                (rows < M)[:, None] & (k < K)[None, :],
                other=0,
            )
            bt = tl.load(
                P + cols[:, None] * K + k[None, :],
                (cols < N)[:, None] & (k < K)[None, :],
                other=0,
            )
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(bt, "dot_pad_only_k")
            acc = tl.dot(a, tl.trans(bt), acc, allow_tf32=False)
        tl.store(
            C + rows[:, None] * N + cols[None, :],
            acc.to(C.dtype.element_ty),
            (rows < M)[:, None] & (cols < N)[None, :],
        )


@triton.jit
def _mm_packed_narrow4_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 16)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, 16), tl.float32)
    for s in tl.range(tl.cdiv(K, BK)):
        k = s * BK + rk
        a = tl.load(
            A + rows[:, None] * K + k[None, :],
            (rows < M)[:, None] & (k < K)[None, :],
            other=0,
        )
        bt = tl.load(
            P + cols[:, None] * K + k[None, :],
            (cols < N)[:, None] & (k < K)[None, :],
            other=0,
        )
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(bt, "dot_pad_only_k")
        acc = tl.dot(a, tl.trans(bt), acc, allow_tf32=False)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc.to(C.dtype.element_ty),
        (rows < M)[:, None] & (cols < N)[None, :],
    )


@triton.jit
def _mm_materialize_broadcast_kernel(
    X,
    Y,
    R: tl.constexpr,
    C: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for pid in range(tl.program_id(0), tl.cdiv(R * C, BLOCK), tl.num_programs(0)):
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        val = tl.load(X + (idx // C) * S0 + (idx % C) * S1, idx < R * C, other=0)
        tl.store(Y + idx, val, idx < R * C)


def _mm_materialize_broadcast(x):
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return y
    with torch_device_fn.device(x.device):
        _mm_materialize_broadcast_kernel[_mm_capped_grid(triton.cdiv(x.numel(), 1024))](
            x,
            y,
            x.shape[0],
            x.shape[1],
            x.stride(0),
            x.stride(1),
            1024,
            multibuffer=False,
        )
    return y


@triton.jit
def _mm_pack_wide_panels_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    for pid in range(tl.program_id(0), (N // BN) * (K // BK), tl.num_programs(0)):
        panel = pid // (K // BK)
        block_k = pid % (K // BK)
        k = block_k * BK + tl.arange(0, BK)
        n = tl.arange(0, BN)
        b = tl.load(B + k[:, None] * N + panel * BN + n[None, :])
        tl.store(P + panel * K * BN + k[:, None] * BN + n[None, :], b)


@triton.jit
def _mm_wide_panel_cube_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    for pid in range(tl.program_id(0), tl.cdiv(M, BM) * (N // BN), tl.num_programs(0)):
        panel = pid % (N // BN)
        rows = (pid // (N // BN)) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for step in range(K // BK):
            k = step * BK + rk
            a = tl.load(A + rows[:, None] * K + k[None, :], rows[:, None] < M, other=0)
            b = tl.load(P + panel * K * BN + k[:, None] * BN + cols[None, :])
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(b, "dot_pad_only_k")
            acc = tl.dot(a, b, acc, allow_tf32=False)
        tl.store(
            C + rows[:, None] * N + panel * BN + cols[None, :],
            acc.to(C.dtype.element_ty),
            rows[:, None] < M,
        )


@triton.jit
def _mm_pack_large_panels_kernel(
    B, P, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    for pid in range(tl.program_id(0), tl.cdiv(N, BN) * (K // BK), tl.num_programs(0)):
        panel = pid // (K // BK)
        block_k = pid % (K // BK)
        k = block_k * BK + tl.arange(0, BK)
        n = tl.arange(0, BN)
        b = tl.load(
            B + k[:, None] * N + panel * BN + n[None, :],
            (panel * BN + n[None, :]) < N,
            other=0,
        )
        tl.store(P + panel * K * BN + k[:, None] * BN + n[None, :], b)


@triton.jit
def _mm_large_panel_cube_kernel(
    A,
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    for pid in range(
        tl.program_id(0), tl.cdiv(M, BM) * tl.cdiv(N, BN), tl.num_programs(0)
    ):
        panel = pid % tl.cdiv(N, BN)
        rows = (pid // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for step in range(K // BK):
            k = step * BK + rk
            a = tl.load(A + rows[:, None] * K + k[None, :], rows[:, None] < M, other=0)
            b = tl.load(P + panel * K * BN + k[:, None] * BN + cols[None, :])
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(b, "dot_pad_only_k")
            acc = tl.dot(a, b, acc, allow_tf32=False)
        tl.store(
            C + rows[:, None] * N + panel * BN + cols[None, :],
            acc.to(C.dtype.element_ty),
            (rows[:, None] < M) & (panel * BN + cols[None, :] < N),
        )


@triton.jit
def _mm_persistent_smallk_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for step in range(4):
        pid = tl.program_id(0) + step * 16
        rows = pid // tl.cdiv(N, BLOCK_N) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid % tl.cdiv(N, BLOCK_N) * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in tl.range(tl.cdiv(K, BLOCK_K)):
            kk = start * BLOCK_K + rk
            a = tl.load(
                A + rows[:, None] * K + kk[None, :],
                mask=(rows < M)[:, None] & (kk < K)[None, :],
                other=0.0,
            )
            b = tl.load(
                B + kk[:, None] * N + cols[None, :],
                mask=(kk < K)[:, None] & (cols < N)[None, :],
                other=0.0,
            )
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(b, "dot_pad_only_k")
            acc = tl.dot(a, b, acc, allow_tf32=False)
        tl.store(
            C + rows[:, None] * N + cols[None, :],
            acc.to(C.dtype.element_ty),
            mask=(rows < M)[:, None] & (cols < N)[None, :],
        )


def _prune_mm_rectangular(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    if is_ascend_910b():
        return prune_cube_tiles_910b(
            configs, args["A"].dtype, ("BLOCK_M", "BLOCK_N", "BLOCK_K")
        )
    m, n = args["M"], args["N"]
    return [
        cfg
        for cfg in configs
        if triton.cdiv(m, cfg.kwargs["BLOCK_M"]) * triton.cdiv(n, cfg.kwargs["BLOCK_N"])
        <= 65535
    ]


def _prune_mm_packed_narrow(configs, named_args, **kwargs):
    if is_ascend_910b():
        args = {**named_args, **kwargs}
        return [
            cfg
            for cfg in configs
            if cube_tile_fits_910b(
                cfg.kwargs["BM"],
                16,
                cfg.kwargs["BK"],
                args["A"].dtype.itemsize,
                cfg.num_stages,
            )
        ]
    m = named_args["M"]
    return [cfg for cfg in configs if triton.cdiv(m, cfg.kwargs["BM"]) <= 65535]


def _prune_mm_narrow_gemv(configs, named_args, **kwargs):
    if is_ascend_910b():
        args = {**named_args, **kwargs}
        return [
            cfg
            for cfg in configs
            if vector_tile_fits_910b(
                cfg.kwargs["BLOCK_OUT"],
                cfg.kwargs["BLOCK_K"],
                args["A"].dtype.itemsize,
                cfg.num_stages,
            )
        ]
    m = named_args["M"]
    return [cfg for cfg in configs if triton.cdiv(m, cfg.kwargs["BLOCK_OUT"]) <= 65535]


mm_rectangular_kernel = make_fixed_gemm_tuner(
    _mm_rectangular_kernel.fn,
    "mm",
    "rectangular",
    ["M", "N", "K"],
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256},
    _prune_mm_rectangular,
    space="mm_rectangular",
)
mm_narrow_gemv_kernel = make_fixed_gemm_tuner(
    _mm_narrow_n_kernel.fn,
    "mm",
    "narrow_gemv",
    ["M", "K"],
    {"BLOCK_OUT": 4, "BLOCK_K": 1024},
    _prune_mm_narrow_gemv,
    space="mm_narrow_gemv",
    stages=1,
)
mm_packed_narrow_kernel = make_fixed_gemm_tuner(
    _mm_packed_narrow_kernel,
    "mm",
    "packed_narrow",
    ["M", "N", "K"],
    {"BM": 64, "BK": 512},
    _prune_mm_packed_narrow,
    space="mm_packed_narrow",
)
mm_packed_narrow_nt_kernel = make_fixed_gemm_tuner(
    _mm_packed_narrow_kernel,
    "mm",
    "packed_narrow_nt",
    ["M", "N", "K"],
    {"BM": 16, "BK": 512},
    _prune_mm_packed_narrow,
    space="mm_packed_narrow",
)
mm_direct_nt_kernel = _make_tiled_tuner("mm", "direct_nt_tiled")
mm_wide_panel_kernel = make_fixed_gemm_tuner(
    _mm_wide_panel_cube_kernel,
    "mm",
    "wide_panels",
    ["M", "N", "K"],
    {"BM": 128, "BK": 128},
    prune_panel_gemm,
    space="panel_gemm",
)
mm_large_panel_kernel = make_fixed_gemm_tuner(
    _mm_large_panel_cube_kernel,
    "mm",
    "large_panels",
    ["M", "N", "K"],
    {"BM": 128, "BK": 256},
    prune_panel_gemm,
    space="panel_gemm",
)


def _launch_mm_packed_narrow(a, packed, out, bm, bk, *, tuner=mm_packed_narrow_kernel):
    m, k = a.shape
    n = out.shape[1]
    options = dict(multibuffer=True, unit_flag=True, sync_solver=False)
    if tune_fixed_gemm(tuner):
        tuner[lambda meta: _mm_capped_grid(triton.cdiv(m, meta["BM"]))](
            a, packed, out, m, n, k, **options
        )
    else:
        _mm_packed_narrow_kernel[_mm_capped_grid(triton.cdiv(m, bm))](
            a, packed, out, m, n, k, bm, bk, **options
        )


def _launch_mm_rectangular(a, b, out, bm, bn, bk):
    m, k = a.shape
    n = b.shape[1]
    options = dict(multibuffer=True, unit_flag=True, sync_solver=False)
    if tune_fixed_gemm(mm_rectangular_kernel):

        def grid(meta):
            return _mm_capped_grid(
                triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"])
            )

        mm_rectangular_kernel[grid](a, b, out, m, n, k, **options)
    else:
        _mm_rectangular_kernel[
            _mm_capped_grid(triton.cdiv(m, bm) * triton.cdiv(n, bn))
        ](a, b, out, m, n, k, bm, bn, bk, **options)


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def _select_mm_nt_route(a, b):
    m, k = a.shape
    n = b.shape[1]
    if (
        a.dtype != b.dtype
        or a.dtype not in (torch.float16, torch.bfloat16)
        or b.is_contiguous()
        or b.stride() != (1, k)
        or min(m, n) == 0
        or max(m * k, n * k, m * n) >= 2**31
    ):
        return None
    if m <= 128 and 64 <= n <= 1024 and k >= 1024:
        return "nt_split"
    if k >= 128 and k % 128 == 0:
        return "nt_dense"
    return None


def _launch_mm_nt(a, b, route):
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    with torch_device_fn.device(a.device), _blockified():
        if route == "nt_split":
            splits = 4
            bm = 16 if m <= 16 else 32
            bn = 64
            grid = (*_mm_capped_grid(triton.cdiv(m, bm) * triton.cdiv(n, bn)), splits)
            partial = torch.empty((splits, m, n), device=a.device, dtype=torch.float32)
            _mm_nt_split_kernel[grid](
                a,
                b,
                partial,
                m,
                n,
                k,
                bm,
                bn,
                128,
                splits,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
            _mm_nt_merge[_mm_capped_grid(triton.cdiv(m * n, 256))](
                partial, c, m * n, splits, 256
            )
        else:
            bm = 16 if m <= 16 else (32 if m <= 32 else 128)
            bn = 16 if n <= 16 else (64 if n <= 64 else 256)
            bk = 256 if k % 256 == 0 else 128
            _mm_nt_kernel[_mm_capped_grid(triton.cdiv(m, bm) * triton.cdiv(n, bn))](
                a,
                b,
                c,
                m,
                n,
                k,
                bm,
                bn,
                bk,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
    return c


def _mm_nn_rectangular_tile(m, n, k):
    # Avoid oversized autotuning tiles for narrow outputs and incomplete M tiles.
    # These ranges retain enough output tiles with a 128x256 Cube tile.
    if k % 128 != 0 or (k == 128 and m % 128 == 0):
        return None
    if 1 <= m <= 16 and n == 64 and 2048 <= k <= 4096:
        return 16, 64, 256
    # Smaller-N workloads already have efficient tuned tiles. Only their
    # unaligned-N, long-K family needs an explicit 128-column tile.
    if 64 <= m <= 128 and 2048 <= n <= 4096:
        if n % 128 != 0 and 4096 <= k <= 8192:
            return 128, 128, 128
        return None
    if (
        (64 <= m <= 512 and 2048 <= n <= 12288 and n % 64 == 0 and 128 <= k <= 8192)
        or (
            1024 <= m <= 4096
            and 128 <= n <= 512
            and 4096 <= k <= 8192
            and (n >= 512 or m >= 2048)
        )
        or (1024 <= m <= 4096 and 1024 <= n <= 4096 and k == 128)
    ):
        return 128, 256, min(256, k)
    return None


def _select_mm_route(a, b):
    """Select the public MM route for the operand shapes and RHS layout."""
    M, K = a.shape
    N = b.shape[1]
    if (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and N == 16
        and 2048 <= K <= 4096
        and K % 1024 == 0
    ):
        return "packed_narrow_nt" if b.T.is_contiguous() else "packed_narrow16"
    if (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and N == 4
        and 8192 < K <= 16384
        and K % 512 == 0
    ):
        return "packed_narrow_nt" if b.T.is_contiguous() else "packed_narrow4"
    nt_route = _select_mm_nt_route(a, b)
    if _select_direct_nt_tile(a, b) is not None and (
        nt_route != "nt_split" or tune_fixed_gemm(mm_direct_nt_kernel)
    ):
        return "direct_nt_tiled"
    if nt_route is not None:
        return nt_route
    if (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and N >= 131072
        and N % 256 == 0
        and 1024 < K <= 2048
        and K % 128 == 0
        and 0 < M <= 128
    ):
        return "wide_panels"
    if (
        a.dtype == b.dtype == torch.bfloat16
        and K % 256 == 0
        and (
            (15360 < M <= 16384 and 2048 < N <= 2304 and 4096 < K <= 8192)
            or (1024 < M <= 2048 and N >= 131072 and 1024 < K <= 2048)
            or (
                1024 < M <= 2048
                and 8192 < N <= 32768
                and N % 256 != 0
                and 2048 < K <= 8192
            )
        )
    ):
        return "large_panels"
    if a.dtype == b.dtype == torch.bfloat16 and (M, N, K) == (480, 4096, 128):
        return "persistent_smallk"
    if a.dtype == b.dtype == torch.bfloat16 and (
        (6144 < N <= 7168 and K == 256 and M > 128)
        or (M, N, K)
        in (
            (200, 7168, 2304),
            (13422, 256, 2048),
            (480, 4608, 7168),
        )
    ):
        return "rectangular_bf16"
    if a.dtype == b.dtype == torch.bfloat16 and (M, N, K) in (
        (1, 64, 4096),
        (6, 64, 4096),
        (16, 64, 2048),
        (4434, 2048, 1024),
        (1352, 3072, 2048),
        (80, 12288, 2048),
        (368, 3072, 2048),
    ):
        return "rectangular_fixed"
    if (
        a.dtype == b.dtype == torch.bfloat16
        and b.is_contiguous()
        and _mm_nn_rectangular_tile(M, N, K) is not None
    ):
        return "rectangular_fixed"
    if M <= 16 and N <= 16 and K < 128:
        return "tiny_reduction"
    if K < 128:
        return "short_reduction"
    if a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16) and K >= 128:
        if N == 1:
            return "narrow_gemv"
        if M <= 32 and N >= 16384:
            return "rectangular_wide"
    return "ascend_gemm"


def mm(a, b):
    logger.debug("GEMS_ASCEND MM")
    device = a.device
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0], "incompatible dimensions"
    if a.shape[0] == 0 or b.shape[1] == 0:
        return torch.empty(
            (a.shape[0], b.shape[1]),
            device=device,
            dtype=get_higher_dtype(a.dtype, b.dtype),
        )
    # Materialize broadcast operands to avoid a zero-stride Cube lowering failure.
    if a.dtype == b.dtype == torch.bfloat16:
        if 0 in a.stride():
            a = _mm_materialize_broadcast(a)
        if 0 in b.stride():
            b = _mm_materialize_broadcast(b)
    if not a.is_contiguous():
        a = a.contiguous()
    M, K = a.shape
    _, N = b.shape
    route = _select_mm_route(a, b)
    if route in ("nt_split", "nt_dense"):
        return _launch_mm_nt(a, b, route)
    if route == "direct_nt_tiled":
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        return _launch_tiled_gemm(
            a, b, c, _select_direct_nt_tile(a, b), tuner=mm_direct_nt_kernel
        )
    if route == "packed_narrow_nt":
        # The NT operand is already stored in the format used by the Cube kernel.
        bm = 64 if N == 16 and M >= 4096 else 16
        bk = 1024 if N == 16 else 512
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        with torch_device_fn.device(device), _blockified():
            _launch_mm_packed_narrow(
                a, b.T, c, bm, bk, tuner=mm_packed_narrow_nt_kernel
            )
        return c
    # Other kernels require row-major or explicitly packed operands.
    if not b.is_contiguous():
        b = b.contiguous()
    # Pack the narrow right operand once so Cube reads contiguous K rows.
    if route == "packed_narrow16":
        block_m = 64 if M >= 4096 else 16
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty((N, K), device=device, dtype=b.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_pack_narrow_b_kernel[_mm_capped_grid(triton.cdiv(K, 512))](
                b,
                packed_b,
                N,
                K,
                512,
                16,
                multibuffer=False,
            )
            _launch_mm_packed_narrow(a, packed_b, c, block_m, 1024)
        return c
    # Deinterleave four BF16 columns into contiguous K vectors for Cube.
    if route == "packed_narrow4":
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty((4, K), device=device, dtype=b.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_pack_narrow4_split_kernel[_mm_capped_grid(K // 512)](
                b,
                packed_b,
                K,
                512,
                multibuffer=False,
            )
            _launch_mm_packed_narrow(a, packed_b, c, 16, 512)
        return c
    # For the very wide BF16 layout, pack B into contiguous N panels so that
    # Cube reads short contiguous rows instead of the original K-strided rows.
    if route == "wide_panels":
        block_m = 16 if M <= 16 else 128
        block_n = 256
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty((N // block_n, K, block_n), device=device, dtype=b.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_pack_wide_panels_kernel[_mm_capped_grid(N // block_n * (K // 64))](
                b, packed_b, N, K, block_n, 64, multibuffer=False
            )
            launch_panel_gemm(
                _mm_wide_panel_cube_kernel,
                mm_wide_panel_kernel,
                (a, packed_b, c),
                (M, N, K),
                block_m,
                128,
            )
        return c
    # Panel families: tall/narrow outputs and wide outputs with a large
    # reduction. K alignment is required by the unmasked panel kernels.
    if route == "large_panels":
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        packed_b = torch.empty(
            (triton.cdiv(N, 256), K, 256), device=device, dtype=b.dtype
        )
        with torch_device_fn.device(device), _blockified():
            _mm_pack_large_panels_kernel[
                _mm_capped_grid(triton.cdiv(N, 256) * (K // 64))
            ](
                b,
                packed_b,
                N,
                K,
                256,
                64,
                multibuffer=False,
            )
            launch_panel_gemm(
                _mm_large_panel_cube_kernel,
                mm_large_panel_kernel,
                (a, packed_b, c),
                (M, N, K),
                128,
                256,
            )
        return c
    # Compatibility exceptions: nearby shapes use the generic path in the
    # established workload. Keep their dispatch until wider ranges are measured.
    # Four M tiles per program keep this 64-tile workload balanced over 16 cores.
    if route == "persistent_smallk":
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        with torch_device_fn.device(device), _blockified():
            _mm_persistent_smallk_kernel[(16,)](
                a,
                b,
                c,
                M,
                N,
                K,
                128,
                256,
                128,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c
    # Wider N tiles reduce program count and K-loop overhead on these layouts.
    if route == "rectangular_bf16":
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        with torch_device_fn.device(device), _blockified():
            _launch_mm_rectangular(a, b, c, 128, 256, min(256, K))
        return c
    # Explicit tiles for narrow outputs and the remaining slow rectangular layouts.
    if route == "rectangular_fixed":
        tile = _mm_nn_rectangular_tile(M, N, K)
        if tile is None:
            tile = (16 if M <= 16 else 128, 64 if N == 64 else 256, 256)
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        with torch_device_fn.device(device), _blockified():
            _launch_mm_rectangular(a, b, c, *tile)
        return c
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    dot_out_dtype = tl.float32

    # Keep tiny reductions off the autotuned Cube path. Some Ascend
    # toolchains select an invalid tile for these partial shapes.
    if route == "tiny_reduction":
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        # Empty reductions still need a nonzero block for tl.arange.
        block_k = min(128, triton.next_power_of_2(max(K, 1)))
        with torch_device_fn.device(device), _blockified():
            _mm_rectangular_kernel[(1,)](
                a,
                b,
                c,
                M,
                N,
                K,
                16,
                16,
                block_k,
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c

    # Keep all short reductions off the autotuned Cube path. Some Ascend
    # toolchains select an invalid tile for partial K tiles.
    if route == "short_reduction":
        block_m = 16 if M <= 16 else 128
        block_n = 32 if N <= 32 else (128 if N <= 128 else 256)
        with torch_device_fn.device(device), _blockified():
            _mm_rectangular_kernel[
                _mm_capped_grid(triton.cdiv(M, block_m) * triton.cdiv(N, block_n))
            ](
                a,
                b,
                c,
                M,
                N,
                K,
                block_m,
                block_n,
                triton.next_power_of_2(max(K, 1)),
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
            )
        return c

    if route == "narrow_gemv":
        with torch_device_fn.device(device):
            if tune_fixed_gemm(mm_narrow_gemv_kernel):
                mm_narrow_gemv_kernel[
                    lambda meta: _mm_capped_grid(triton.cdiv(M, meta["BLOCK_OUT"]))
                ](a, b, c, M, N, K)
            else:
                _mm_narrow_n_kernel[_mm_capped_grid(triton.cdiv(M, 4))](
                    a,
                    b,
                    c,
                    M,
                    N,
                    K,
                    BLOCK_OUT=4,
                    BLOCK_K=min(2048, triton.next_power_of_2(K)),
                )
        return c
    if route == "rectangular_wide":
        block_m = 16 if M <= 16 else 32
        block_n = 256
        with torch_device_fn.device(device), _blockified():
            _launch_mm_rectangular(a, b, c, block_m, block_n, 256)
        return c

    # launch kernel
    grid = lambda META: _mm_grid(M, N, META)
    with torch_device_fn.device(a.device), _blockified():
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype,
            GROUP_M=1,
            multibuffer=True,
            unit_flag=True,
            sync_solver=False,
        )
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS_ASCEND MM_OUT")
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0], "incompatible dimensions"
    if out.device != a.device or b.device != a.device:
        raise RuntimeError("all tensors must be on the same device")
    if out.dtype != get_higher_dtype(a.dtype, b.dtype):
        raise RuntimeError("out dtype must match the result dtype")
    # Save aliased operands before resizing or writing the destination.
    if torch._C._is_alias_of(out, a):
        a = copy_tensor(a, a)
    if torch._C._is_alias_of(out, b):
        b = copy_tensor(b, b)
    if out.shape != (a.shape[0], b.shape[1]):
        out.resize_(a.shape[0], b.shape[1])
    if out.numel() == 0:
        return out
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    dot_out_dtype = tl.float32
    grid = lambda META: _mm_grid(M, N, META)
    with torch_device_fn.device(a.device), _blockified():
        mm_kernel[grid](
            a,
            b,
            out,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            dot_out_dtype=dot_out_dtype,
            GROUP_M=8,
        )
    return out
