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
import os
from contextlib import contextmanager
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import LibTuner, LibTunerRunMode

logger = logging.getLogger(__name__)


_BMM_CONFIGS = runtime.get_tuned_config("bmm")
_BMM_EXPAND_PATH = str(
    Path(__file__).resolve().parent.parent / "bmm_ascend_expand.yaml"
)
FIXED_GEMM_EXPAND_PATH = str(
    Path(__file__).parent.parent / "fixed_gemm_ascend_expand.yaml"
)

# CANN 9.1 platform_config/Ascend910B4-1.ini, AICoreSpec.
_L0B_BYTES = 64 * 1024
_L0C_BYTES = 128 * 1024
_L1_BYTES = 512 * 1024
_UB_BYTES = 192 * 1024
_FP32_BYTES = 4


def is_ascend_910b():
    arch = triton.runtime.driver.active.get_current_target().arch
    # The bare Ascend910B target uses AIC-100 and different memory sizes.
    return str(arch).startswith(
        ("Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4")
    )


def is_ascend_910c():
    target = triton.runtime.driver.active.get_current_target()
    return str(target.arch).startswith("Ascend910_93")


def cube_tile_fits_910b(bm, bn, bk, element_bytes, stages):
    """Conservatively budget the FP32 accumulator and staged L1 input panels."""
    # CANN can subdivide the L1 panels when issuing L0A/L0B loads.
    return (
        bm * bn * _FP32_BYTES <= _L0C_BYTES
        and (bm + bn) * bk * element_bytes * max(stages, 1) <= _L1_BYTES
    )


def prune_cube_tiles_910b(configs, dtype, names):
    return [
        config
        for config in configs
        if cube_tile_fits_910b(
            *(config.kwargs[name] for name in names),
            dtype.itemsize,
            config.num_stages,
        )
    ]


def vector_tile_fits_910b(rows, reduction, element_bytes, stages):
    """Budget FP32 working tiles and staged inputs in Unified Buffer."""
    staged_bytes = max(stages, 1) * element_bytes
    matrix_bytes = rows * reduction * (2 * _FP32_BYTES + staged_bytes)
    vector_bytes = reduction * (_FP32_BYTES + staged_bytes) + rows * _FP32_BYTES
    return matrix_bytes + vector_bytes <= _UB_BYTES


class AscendBlasTuner(LibTuner.get("flagtune")):
    def get_key(self, args):
        key = super().get_key(args)
        if is_ascend_910b():
            # Rerank old cache entries under the 910B rules. Exact dimensions
            # matter for padding and launch limits even with a bucketed strategy.
            key += (
                "ascend910b_blas_resources_v1",
                *(args[k] for k in self.keys if k in args),
            )
        return key


@contextmanager
def _blockified():
    key = "TRITON_ALL_BLOCKS_PARALLEL"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def tune_fixed_gemm(tuner):
    # Keep the upstream preset until tuning is selected or requested offline.
    autotuner = getattr(tuner, "fn", None)
    if not isinstance(autotuner, LibTuner):
        return False
    return (
        autotuner._run_mode is not LibTunerRunMode.NORMAL
        or runtime.flagtune_enabled(autotuner._flagtune_op_name)
        or runtime.resolve_cost_model_intent(supports_cost_model=True)
        is runtime.CostModelIntent.REQUIRED
    )


def gemm_dtype_resolver(op, rhs="B"):
    """Match the public input/output dtype order and omit scratch pointers."""
    names = ("A", rhs, "Out") if op in {"mm", "bmm"} else ("Bias", "A", rhs, "Out")

    def resolve(arguments):
        return tuple(arguments[name].dtype for name in names)

    return resolve


def _prune_bmm_configs(configs, named_args, **kwargs):
    # Keep only structural constraints: arange extents must be powers of two,
    # and each A/B/accumulator tile must fit Triton's tensor-element limit.
    # CANN resource allocation depends on the compiled pipeline; let the
    # compiler reject resource failures instead of guessing a UB/L0 budget.
    pruned = []
    for config in configs:
        tile_m = config.kwargs["TILE_M"]
        tile_n = config.kwargs["TILE_N"]
        tile_k = config.kwargs["TILE_K"]
        if not all(
            size > 0 and size & (size - 1) == 0 for size in (tile_m, tile_n, tile_k)
        ):
            continue
        if config.kwargs["GROUP_M"] < 1:
            continue
        if max(tile_m * tile_k, tile_k * tile_n, tile_m * tile_n) > 1048576:
            continue
        pruned.append(config)
    if not pruned:
        raise RuntimeError("No valid BMM configurations after pruning")
    return pruned


class ResourcePrunedGemmTuner(AscendBlasTuner):
    """Use hardware constraints for search and the existing policy for Default."""

    tile_prefix = "TILE"
    prune_910c_bf16 = None

    def __init__(self, *args, **kwargs):
        kwargs["prune_configs_by"] = {"early_config_prune": self._prune_candidates}
        super().__init__(*args, **kwargs)

    def _prune_candidates(self, configs, named_args, **kwargs):
        args = {**named_args, **kwargs}
        is_910b = is_ascend_910b()
        names = tuple(f"{self.tile_prefix}_{axis}" for axis in ("M", "N", "K"))
        if (
            self._flagtune_mode is runtime.TuningMode.DEFAULT
            and getattr(self, "_run_mode", None)
            is not LibTunerRunMode.EXHAUSTIVE_COLLECTION
        ):
            preferred = self.default_prune(configs, named_args, **kwargs)
            if not is_910b:
                return preferred
            dtype = args["A"].dtype
            return prune_cube_tiles_910b(
                preferred, dtype, names
            ) or prune_cube_tiles_910b(configs, dtype, names)

        is_910c = is_ascend_910c()
        prune = self.prune_910c_bf16 if is_910c else None
        if prune is not None:
            if args["A"].dtype != torch.bfloat16:
                prune = None
        valid = []
        for config in configs:
            meta = config.kwargs
            bm, bn, bk = (meta[name] for name in names)
            if not all(tile > 0 and tile & (tile - 1) == 0 for tile in (bm, bn, bk)):
                continue
            if meta.get("GROUP_M", 1) < 1 or meta.get("SPLIT_K", 1) != 1:
                continue
            if max(bm * bn, bm * bk, bn * bk) > 1048576:
                continue
            if is_910c and bm * bn > 32768:
                continue
            if is_910b and not cube_tile_fits_910b(
                bm, bn, bk, args["A"].dtype.itemsize, config.num_stages
            ):
                continue
            if prune is not None and prune(args["K"], bk, config.num_stages):
                continue
            valid.append(config)
        return valid


class _BMMFlagTuneTuner(ResourcePrunedGemmTuner):
    default_prune = staticmethod(_prune_bmm_configs)


@libentry()
@libtuner(
    configs=_BMM_CONFIGS,
    key=["M", "N", "K", "BATCH", "DOT_PAD_ONLY_K"],
    strategy=["default"] * 5,
    warmup=5,
    rep=5,
    use_cuda_graph=True,
    flagtune_op_name="bmm",
    flagtune_expand_op_name="bmm_ascend",
    flagtune_yaml_path=_BMM_EXPAND_PATH,
    policy=_BMMFlagTuneTuner,
    flagtune_op_id="flaggems/bmm",
    flagtune_variant="batched_gemm",
)
@triton.heuristics(_hcu.HEURISTICS_CONFIGS["bmm"])
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    BATCH: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DOT_PAD_ONLY_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    # batch offsets
    pid_b = ext.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)
    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    num_iters = tl.cdiv(K, TILE_K)
    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for i in range(num_iters):
        if DIVISIBLE_K:
            mask_a = None if DIVISIBLE_M else mask_m[:, None]
            mask_b = None if DIVISIBLE_N else mask_n[None, :]
        else:
            mask_k = offs_k < K - i * TILE_K
            mask_a = (
                mask_k[None, :] if DIVISIBLE_M else mask_m[:, None] & mask_k[None, :]
            )
            mask_b = (
                mask_k[:, None] if DIVISIBLE_N else mask_k[:, None] & mask_n[None, :]
            )
        if DIVISIBLE_K and DIVISIBLE_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        if DIVISIBLE_K and DIVISIBLE_N:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

        if DOT_PAD_ONLY_K:
            extension.compile_hint(a, "dot_pad_only_k")
            extension.compile_hint(b, "dot_pad_only_k")
        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M:
        mask_c = mask_n[None, :]
    elif DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    if DIVISIBLE_M and DIVISIBLE_N:
        tl.store(o_ptrs, o)
    else:
        tl.store(o_ptrs, o, mask_c)


def bmm(A, B):
    logger.debug("GEMS_ASCEND BMM")
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
    route, tile = _select_bmm_route(A, B)
    if route == "direct_nt":
        return _launch_nt_gemm(A, B, out, tuner=bmm_tiled_kernel)
    if route == "direct_nt_tiled":
        return _launch_tiled_gemm(A, B, out, tile, tuner=bmm_direct_nt_kernel)
    if route == "column_gemv":
        return _launch_column_gemv(A, B, out, tuner=bmm_column_gemv_kernel)
    if route == "narrow_gemv":
        return _launch_narrow_gemv(A, B, out, tuner=bmm_narrow_gemv_kernel)
    if route == "packed_rhs":
        return _launch_narrow_pack(A, B, out)
    if tile is not None:
        return _launch_tiled_gemm(A, B, out, tile, tuner=bmm_tiled_kernel)
    B = B.contiguous()
    dot_pad_only_k = (
        A.dtype == torch.bfloat16
        and batch == 1
        and M >= 256
        and N >= 128
        and N % 16 == 0
        and K >= 128
    )

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )

    with torch_device_fn.device(A.device), _blockified():
        bmm_kernel[grid_fn](
            A,
            B,
            out,
            M,
            N,
            K,
            BATCH=batch,
            DOT_PAD_ONLY_K=dot_pad_only_k,
            enable_auto_blockify=True,
        )
    return out


def _select_bmm_route(a, b):
    if _use_nt_gemm(a, b):
        return "direct_nt", None
    if _use_column_gemv(a, b):
        return "column_gemv", None
    if _use_narrow_gemv(a, b):
        return "narrow_gemv", None
    if _use_narrow_pack(a, b):
        return "packed_rhs", None
    tile = _select_rectangular_tile(a, b)
    if tile is not None:
        return ("split_k" if tile[3] > 1 else "fixed_tile"), tile
    tile = _select_direct_nt_tile(a, b)
    if tile is not None:
        return "direct_nt_tiled", tile
    return "batched_gemm", None


# Shared Ascend matmul fast paths for addmm, bmm, and baddbmm.
def _can_use_dense_bf16_gemm(a, b):
    # Public callers materialize the LHS before dispatching either fast path.
    m, k = a.shape[-2:]
    n = b.shape[-1]
    batch = a.shape[0] if a.ndim == 3 else 1
    return (
        a.dtype == b.dtype == torch.bfloat16
        and a.shape[:-2] == b.shape[:-2]
        and b.shape[-2] == k
        and min(batch, m, n, k) > 0
        and batch * max(m * k, n * k, m * n) < 2**31
    )


def _select_direct_nt_tile(a, b):
    """Read aligned dense transposed RHS tiles directly from K-contiguous storage."""
    m, k = a.shape[-2:]
    n = b.shape[-1]
    if (
        n < 4
        or (n != 4 and n % 16)
        or k < 128
        or k % 128
        or b.is_contiguous()
        or not b.transpose(-2, -1).is_contiguous()
        or not _can_use_dense_bf16_gemm(a, b)
    ):
        return None
    batch = a.shape[0] if a.ndim == 3 else 1
    bm = 16 if m <= 16 else 32 if m <= 64 else 64 if m <= 256 else 128
    bn = min(256, max(16, triton.next_power_of_2(n)))
    bk = 128 if k == 128 else 256
    if n <= 512 and k >= 4096:
        bm, bn, bk = min(bm, 64), min(bn, 128), 512
    if batch * triton.cdiv(m, bm) * triton.cdiv(n, bn) > 65535:
        return None
    return bm, bn, bk, 1, False


def _use_nt_gemm(a, b):
    m, k = a.shape[-2:]
    n = b.shape[-1]
    return (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and a.is_contiguous()
        and not b.is_contiguous()
        and b.transpose(-2, -1).is_contiguous()
        and a.shape[:-2] == b.shape[:-2]
        and (a.ndim == 2 or a.shape[0] > 0)
        and b.shape[-2] == k
        and 0 < m <= 512
        and 4 <= n <= 8192
        and n % 4 == 0
        and k >= 256
        and k % 256 == 0
    )


def _select_rectangular_tile(a, b, *, rectangular_nn=True):
    m, k = a.shape[-2:]
    n = b.shape[-1]
    nt = not b.is_contiguous()
    if (
        a.dtype != b.dtype
        or a.dtype not in (torch.float16, torch.bfloat16)
        or not a.is_contiguous()
        or (nt and not b.transpose(-2, -1).is_contiguous())
        or a.shape[:-2] != b.shape[:-2]
        or (a.ndim == 3 and a.shape[0] == 0)
        or b.shape[-2] != k
    ):
        return None
    if nt and k == 64 and 64 <= m <= 256 and 64 <= n <= 256:
        return 64, 64, 64, 1, False
    if 32 <= m <= 128 and 64 <= n <= 512 and 1024 <= k <= 4096 and (k % 16 or n % 16):
        # Larger batches already supply enough work to use a taller NN tile.
        bm = (
            64
            if rectangular_nn and not nt and a.ndim == 3 and a.shape[0] >= 8 and m > 64
            else 32
        )
        return bm, 64, 256, 1, False
    if nt:
        return None
    if 0 < m <= 128 and 32 <= n <= 128 and n % 32 == 0 and k in (2048, 4096):
        return 16, 64, 256, 4, True
    if rectangular_nn and 16 <= m <= 128 and 128 < n <= 512 and n % 16 == 0:
        if 2048 <= k <= 8192 and k % 512 == 0:
            return 32, 64, 512, 1, True
    if rectangular_nn and 64 <= m <= 128 and 8192 <= n <= 32768 and n % 256 == 0:
        if 4096 <= k <= 8192 and k % 128 == 0:
            return 64, 256, 128, 1, True
    return None


@triton.jit
def _tiled_gemm_kernel(
    A,
    B,
    Bias,
    Out,
    Scratch,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    WITH_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    HINT: tl.constexpr,
):
    pid = tl.program_id(0)
    split = pid % SPLIT
    tile = pid // SPLIT
    batch = tl.program_id(1)
    m = (tile // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    n = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    k0 = tl.arange(0, BK)
    a_base = A + batch * M * K
    b_base = B + batch * K * N
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK * SPLIT)):
        k = (start * SPLIT + split) * BK + k0
        valid_k = k < K if K % (BK * SPLIT) else tl.full((BK,), True, tl.int1)
        a = tl.load(
            a_base + m[:, None] * K + k[None, :], (m[:, None] < M) & valid_k[None, :], 0
        )
        if TRANSPOSED:
            # The transposed view stores consecutive K elements for each column.
            b = tl.load(
                b_base + n[:, None] * K + k[None, :],
                (n[:, None] < N) & valid_k[None, :],
                0,
            ).T
        else:
            b = tl.load(
                b_base + k[:, None] * N + n[None, :],
                valid_k[:, None] & (n[None, :] < N),
                0,
            )
        if HINT:
            extension.compile_hint(a, "dot_pad_only_k")
            extension.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc, allow_tf32=False)
    offset = batch * M * N + m[:, None] * N + n[None, :]
    valid = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT > 1:
        tl.store(Scratch + split * BATCH * M * N + offset, acc, valid)
    else:
        if WITH_BIAS:
            acc += tl.load(Bias + offset, valid, 0).to(tl.float32)
        tl.store(Out + offset, acc, valid)


@triton.jit
def _finish_split_gemm(
    Scratch,
    Bias,
    Out,
    TOTAL: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
    WITH_BIAS: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLIT)
    parts = tl.load(
        Scratch + split[:, None] * TOTAL + offset[None, :],
        offset[None, :] < TOTAL,
        0,
    )
    value = tl.sum(parts, axis=0)
    if WITH_BIAS:
        value += tl.load(Bias + offset, offset < TOTAL, 0).to(tl.float32)
    tl.store(Out + offset, value, offset < TOTAL)


_TILED_EXPAND_PATH = str(
    Path(__file__).resolve().parent.parent / "tiled_gemm_ascend_expand.yaml"
)
_TILED_KEY = ["BATCH", "M", "N", "K", "TRANSPOSED", "WITH_BIAS", "HINT"]
_NT_BIAS_TAIL_MAX_BK_910B = 128


def prune_910c_nt_column_tail(n, bn):
    """Reject wide masked column tiles for BF16 transposed RHS on 910C."""
    return bn >= 512 and n % bn != 0


def _nt_bias_tail_tile_fits_910b(n, bn, bk):
    # BF16 NT GEMM with fused bias can stall on 910B when a wide reduction
    # tile is combined with a masked column tail.
    return n % bn == 0 or bk <= _NT_BIAS_TAIL_MAX_BK_910B


def masked_bias_rhs_fits_910b(n, k, bn, bk, element_bytes):
    """Leave room for padding when a transposed BF16 RHS has two masked axes."""
    # A full L0B-sized tile with this layout stalled the bias kernel on 910B4-1.
    return n % bn == 0 or k >= bk or bn * bk * element_bytes < _L0B_BYTES


def _prune_tiled_gemm_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = (args[key] for key in ("M", "N", "K"))
    batch = args["BATCH"]
    nt_bf16 = args["TRANSPOSED"] and args["A"].dtype == torch.bfloat16
    is_910b = is_ascend_910b()
    prune_nt_tail = nt_bf16 and is_ascend_910c()
    # A large RHS tile with padding in both axes can fault in the bias path.
    prune_bias_tail = nt_bf16 and args["WITH_BIAS"]
    return [
        c
        for c in configs
        if c.kwargs["BM"] * c.kwargs["BN"] <= 32768
        and batch * triton.cdiv(m, c.kwargs["BM"]) * triton.cdiv(n, c.kwargs["BN"])
        <= 65535
        and (not args["HINT"] or k % c.kwargs["BK"] == 0)
        and not (prune_nt_tail and prune_910c_nt_column_tail(n, c.kwargs["BN"]))
        and (
            not (is_910b and prune_bias_tail)
            or _nt_bias_tail_tile_fits_910b(n, c.kwargs["BN"], c.kwargs["BK"])
        )
        and (
            not is_910b
            or cube_tile_fits_910b(
                c.kwargs["BM"],
                c.kwargs["BN"],
                c.kwargs["BK"],
                args["A"].dtype.itemsize,
                c.num_stages,
            )
        )
        and not (
            prune_bias_tail
            and not masked_bias_rhs_fits_910b(
                n, k, c.kwargs["BN"], c.kwargs["BK"], args["A"].dtype.itemsize
            )
        )
    ]


def _prune_direct_nt_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    n, k = args["N"], args["K"]
    return [
        config
        for config in _prune_tiled_gemm_configs(configs, named_args, **kwargs)
        if k % config.kwargs["BK"] == 0
        and config.kwargs["BN"] <= min(256, max(16, triton.next_power_of_2(n)))
        and max(config.kwargs["BM"], config.kwargs["BN"]) * config.kwargs["BK"] <= 65536
    ]


class _TiledGemmTuner(AscendBlasTuner):
    def get_key(self, args):
        key = super().get_key(args)
        if (
            args["TRANSPOSED"]
            and args["WITH_BIAS"]
            and args["A"].dtype == torch.bfloat16
            and is_ascend_910b()
        ):
            # Cached winners from the old search space can bypass pruning.
            return (*key, "ascend910b_nt_bias_tail_v1")
        return key

    def _flagtune_configs_for_mode(self, op_name, mode):
        configs, strategy = super()._flagtune_configs_for_mode(op_name, mode)
        if mode is runtime.TuningMode.DEFAULT and is_ascend_910b():
            # The small default tile alone cannot cover narrow NT layouts
            # and large grids. The fallback tiles are also in Expanded.
            configs = configs + [
                triton.Config(dict(BM=bm, BN=bn, BK=bk), num_warps=4, num_stages=1)
                for bm, bn, bk in (
                    (16, 16, 64),
                    (16, 32, 64),
                    (64, 64, 64),
                    (128, 128, 128),
                    (128, 256, 128),
                )
            ]
        return configs, strategy


def _make_bmm_tiled_tuner(variant, prune):
    return libentry()(
        libtuner(
            configs=[
                triton.Config(dict(BM=16, BN=64, BK=128), num_warps=4, num_stages=2)
            ],
            key=_TILED_KEY,
            prune_configs_by={"early_config_prune": prune},
            use_cuda_graph=True,
            policy=_TiledGemmTuner,
            flagtune_op_name="bmm",
            flagtune_op_id="flaggems/bmm",
            flagtune_variant=variant,
            flagtune_expand_op_name="tiled_gemm_ascend",
            flagtune_yaml_path=_TILED_EXPAND_PATH,
            flagtune_dtype_resolver=gemm_dtype_resolver("bmm"),
        )(_tiled_gemm_kernel)
    )


bmm_tiled_kernel = _make_bmm_tiled_tuner("tiled_gemm", _prune_tiled_gemm_configs)
bmm_direct_nt_kernel = _make_bmm_tiled_tuner(
    "direct_nt_tiled", _prune_direct_nt_configs
)


def _launch_tiled_gemm(a, b, out, tile, bias=None, *, tuner=None):
    bm, bn, bk, split, hint = tile
    m, k = a.shape[-2:]
    n = b.shape[-1]
    batch = a.shape[0] if a.ndim == 3 else 1
    total = batch * m * n
    transposed = not b.is_contiguous()
    with_bias = bias is not None
    bias_arg = a if bias is None else bias
    partial = (
        torch.empty((split, total), device=a.device, dtype=torch.float32)
        if split > 1
        else out
    )
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn) * split, batch)
    # The original launcher's preset is the default for these fast paths.
    # Only opt-in FlagTune or an explicit offline run can replace that preset.
    tune = split == 1 and tuner is not None and tune_fixed_gemm(tuner)
    with torch_device_fn.device(a.device):
        with _blockified():
            launch_args = (
                a,
                b,
                bias_arg,
                out,
                partial,
                batch,
                m,
                n,
                k,
                transposed,
                with_bias,
            )
            launch_options = dict(
                multibuffer=True,
                unit_flag=True,
                sync_solver=False,
                enable_auto_blockify=True,
            )
            if tune:

                def tuned_grid(meta):
                    return (
                        triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),
                        batch,
                    )

                tuner[tuned_grid](*launch_args, SPLIT=1, HINT=hint, **launch_options)
            else:
                if (
                    split == 1
                    and with_bias
                    and transposed
                    and a.dtype == torch.bfloat16
                    and not _nt_bias_tail_tile_fits_910b(n, bn, bk)
                    and is_ascend_910b()
                ):
                    bk = _NT_BIAS_TAIL_MAX_BK_910B
                _tiled_gemm_kernel[grid](
                    *launch_args, bm, bn, bk, split, hint, **launch_options
                )
        if split > 1:
            _finish_split_gemm[(triton.cdiv(total, 256),)](
                partial, bias_arg, out, total, split, 256, bias is not None
            )
    return out


def _launch_nt_gemm(a, b, out, bias=None, block_k=256, *, tuner=None):
    m = a.shape[-2]
    n = b.shape[-1]
    if m >= 256 and n >= 128 and m % 128 == 0 and n % 128 == 0:
        bm = bn = 128
    else:
        bm = 16 if m <= 64 else 64
        bn = 16 if n <= 16 else 64
    return _launch_tiled_gemm(a, b, out, (bm, bn, block_k, 1, False), bias, tuner=tuner)


def _use_narrow_gemv(a, b, *, materialize_rhs=False):
    m, k = a.shape[-2:]
    return (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and b.shape[-2:] == (k, 1)
        and a.shape[:-2] == b.shape[:-2]
        and (a.ndim == 2 or a.shape[0] > 0)
        and a.is_contiguous()
        and (materialize_rhs or b.is_contiguous())
        and m >= 256
        and m % 8 == 0
        and k >= 1024
        and k % 1024 == 0
    )


def _column_gemv_tile(batch, m, k):
    if batch * m >= 1024:
        return 32, 512
    if batch * m >= 256 and m >= 16:
        return 16, 1024
    return min(8, triton.next_power_of_2(m)), min(2048, triton.next_power_of_2(k))


def _use_column_gemv(a, b):
    m, k = a.shape[-2:]
    if b.shape[-1] != 1 or not b.is_contiguous() or not _can_use_dense_bf16_gemm(a, b):
        return False
    batch = a.shape[0]
    bm, _ = _column_gemv_tile(batch, m, k)
    return batch * triton.cdiv(m, bm) <= 65535


@triton.jit
def _column_gemv_kernel(
    A,
    B,
    Bias,
    Out,
    M: tl.constexpr,
    K: tl.constexpr,
    WITH_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BATCH: tl.constexpr,
):
    tiles: tl.constexpr = tl.cdiv(M, BM)
    pid = tl.program_id(0)
    batch = pid // tiles
    rows = pid % tiles * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        a = tl.load(
            A + (batch * M + rows[:, None]) * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            0,
        ).to(tl.float32)
        b = tl.load(B + batch * K + k, k < K, 0).to(tl.float32)
        acc += a * b[None, :]
    result = tl.sum(acc, axis=1)
    offset = batch * M + rows
    if WITH_BIAS:
        result += tl.load(Bias + offset, rows < M, 0).to(tl.float32)
    tl.store(Out + offset, result, rows < M)


def _prune_column_gemv(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    is_910b = is_ascend_910b()
    return [
        config
        for config in configs
        if config.kwargs["BM"] * config.kwargs["BK"] <= 16384
        and args["BATCH"] * triton.cdiv(args["M"], config.kwargs["BM"]) <= 65535
        and (
            not is_910b
            or vector_tile_fits_910b(
                config.kwargs["BM"],
                config.kwargs["BK"],
                args["A"].dtype.itemsize,
                config.num_stages,
            )
        )
    ]


def _make_bmm_gemv_tuner(kernel, variant, prune):
    # BATCH is launch metadata and keeps distinct batch sizes in the tuning key.
    return libentry()(
        libtuner(
            configs=[triton.Config({"BM": 8, "BK": 1024}, num_warps=4, num_stages=1)],
            key=["M", "K", "WITH_BIAS", "BATCH"],
            prune_configs_by={"early_config_prune": prune},
            use_cuda_graph=True,
            policy=AscendBlasTuner,
            flagtune_op_name="bmm",
            flagtune_op_id="flaggems/bmm",
            flagtune_variant=variant,
            flagtune_expand_op_name=f"{variant}_ascend",
            flagtune_yaml_path=FIXED_GEMM_EXPAND_PATH,
            flagtune_dtype_resolver=gemm_dtype_resolver("bmm"),
            disable_aabs=True,
        )(kernel)
    )


bmm_column_gemv_kernel = _make_bmm_gemv_tuner(
    _column_gemv_kernel, "column_gemv", _prune_column_gemv
)


def _launch_column_gemv(a, b, out, bias=None, *, tuner):
    batch, m, k = a.shape
    bm, bk = _column_gemv_tile(batch, m, k)
    tiles = triton.cdiv(m, bm)
    args = (a, b, a if bias is None else bias, out, m, k, bias is not None)
    with torch_device_fn.device(a.device):
        if tune_fixed_gemm(tuner):
            tuner[lambda meta: (batch * triton.cdiv(m, meta["BM"]),)](
                *args, BATCH=batch
            )
        else:
            _column_gemv_kernel[(batch * tiles,)](*args, bm, bk, batch)
    return out


@triton.jit
def _narrow_gemv_kernel(
    A,
    B,
    Bias,
    Out,
    M: tl.constexpr,
    K: tl.constexpr,
    WITH_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BATCH: tl.constexpr = 1,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    batch = tl.program_id(1)
    rk = tl.arange(0, BK)
    a_base = A + (batch * M + rows[:, None]) * K
    b_base = B + batch * K
    acc = tl.zeros((BM, BK), tl.float32)
    for start in range(K // BK):
        k = start * BK + rk
        a = tl.load(a_base + k[None, :]).to(tl.float32)
        b = tl.load(b_base + k).to(tl.float32)
        acc += a * b[None, :]
    result = tl.sum(acc, axis=1)
    offset = batch * M + rows
    if WITH_BIAS:
        result += tl.load(Bias + offset).to(tl.float32)
    tl.store(Out + offset, result)


def _prune_narrow_gemv(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, k, batch = (args[key] for key in ("M", "K", "BATCH"))
    is_910b = is_ascend_910b()
    # The original Vector kernel has unmasked loads and uses K // BK.
    return [
        cfg
        for cfg in configs
        if m % cfg.kwargs["BM"] == 0
        and k % cfg.kwargs["BK"] == 0
        and batch * (m // cfg.kwargs["BM"]) <= 65535
        and (
            not is_910b
            or vector_tile_fits_910b(
                cfg.kwargs["BM"],
                cfg.kwargs["BK"],
                args["A"].dtype.itemsize,
                cfg.num_stages,
            )
        )
    ]


bmm_narrow_gemv_kernel = _make_bmm_gemv_tuner(
    _narrow_gemv_kernel, "narrow_gemv", _prune_narrow_gemv
)


def _launch_narrow_gemv(a, b, out, bias=None, *, tuner):
    m, k = a.shape[-2:]
    batch = a.shape[0] if a.ndim == 3 else 1
    args = (a, b, a if bias is None else bias, out, m, k, bias is not None)
    with torch_device_fn.device(a.device):
        if tune_fixed_gemm(tuner):
            tuner[lambda meta: (m // meta["BM"], batch)](*args, BATCH=batch)
        else:
            _narrow_gemv_kernel[(m // 8, batch)](*args, 8, 1024)
    return out


def _use_narrow_pack(a, b, *, materialize_rhs=False):
    m, k = a.shape[-2:]
    n = b.shape[-1]
    return (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and a.is_contiguous()
        and (materialize_rhs or b.is_contiguous())
        and a.shape[:-2] == b.shape[:-2]
        and (a.ndim == 2 or a.shape[0] > 0)
        and b.shape[-2] == k
        and 0 < m <= 64
        and k % 512 == 0
        and ((n == 4 and 8192 <= k <= 16384) or (n == 16 and 2048 <= k <= 4096))
    )


@triton.jit
def _pack_narrow_rhs(B, Packed, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    start = tl.program_id(0) * BK
    batch = tl.program_id(1)
    B += batch * N * K
    Packed += batch * N * K
    k = start + tl.arange(0, BK)
    if N == 4:
        # Deinterleave the four columns after one contiguous load.
        b = tl.load(B + start * 4 + tl.arange(0, BK * 4)).reshape((BK, 2, 2))
        b02, b13 = tl.split(b)
        b0, b2 = tl.split(b02)
        b1, b3 = tl.split(b13)
        tl.store(Packed + k, b0)
        tl.store(Packed + K + k, b1)
        tl.store(Packed + 2 * K + k, b2)
        tl.store(Packed + 3 * K + k, b3)
    else:
        n = tl.arange(0, N)
        b = tl.load(B + k[:, None] * N + n[None, :])
        tl.store(Packed + n[:, None] * K + k[None, :], b.T)


def _launch_narrow_pack(a, b, out, bias=None):
    k, n = b.shape[-2:]
    batch = b.shape[0] if b.ndim == 3 else 1
    packed = torch.empty((*b.shape[:-2], n, k), device=b.device, dtype=b.dtype)
    with torch_device_fn.device(b.device):
        _pack_narrow_rhs[(k // 512, batch)](b, packed, n, k, 512, multibuffer=False)
    return _launch_nt_gemm(a, packed.transpose(-2, -1), out, bias, block_k=512)
