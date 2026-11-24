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
import triton.language.extra.cann.extension as extension

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils import broadcastable_to, libentry, libtuner

from .bmm import (
    ResourcePrunedGemmTuner,
    _blockified,
    _launch_narrow_gemv,
    _launch_narrow_pack,
    _launch_nt_gemm,
    _launch_tiled_gemm,
    _make_gemv_tuner,
    _make_split_gemm_tuner,
    _make_tiled_gemm_tuner,
    _narrow_gemv_kernel,
    _prune_narrow_gemv,
    _prune_tiled_gemm_configs,
    _select_rectangular_tile,
    _use_narrow_gemv,
    _use_narrow_pack,
    _use_nt_gemm,
    gemm_dtype_resolver,
    is_ascend_910c,
)
from .mm import _mm_pack_wide_panels_kernel, launch_panel_gemm, prune_panel_gemm

logger = logging.getLogger(__name__)
_MAX_ADDMM_PROGRAMS = 32768
_ADDMM_EXPAND_PATH = str(
    Path(__file__).resolve().parent.parent / "addmm_ascend_expand.yaml"
)

addmm_tiled_kernel = _make_tiled_gemm_tuner(
    "addmm", "tiled_gemm", _prune_tiled_gemm_configs, yaml_path=_ADDMM_EXPAND_PATH
)
addmm_split_kernel = _make_split_gemm_tuner(
    op_name="addmm", yaml_path=_ADDMM_EXPAND_PATH
)
addmm_narrow_gemv_kernel = _make_gemv_tuner(
    _narrow_gemv_kernel,
    "addmm",
    "narrow_gemv",
    _prune_narrow_gemv,
    yaml_path=_ADDMM_EXPAND_PATH,
)


# Keep addmm's validated fallback independent of the shared mm tuning space.
# In particular, NT layouts and small GEMMs cannot use every mm candidate.
_ADDMM_BASE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k, "SPLIT_K": 1},
        num_warps=4,
        num_stages=2,
    )
    for m, n, k in ((16, 32, 32), (128, 128, 128), (256, 256, 256))
]
_ADDMM_HEURISTICS = {
    **_hcu.HEURISTICS_CONFIGS["mm"],
    "DIVISIBLE_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
    "DIVISIBLE_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
}


def _prune_addmm_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    if (
        args["A"].dtype not in (torch.float16, torch.float32, torch.bfloat16)
        or args["stride_ak"] != 1
        or args["stride_bn"] != 1
        or args["stride_cn"] != 1
        or min(args["M"], args["N"]) < 128
    ):
        return _ADDMM_BASE_CONFIGS
    if len(configs) > 7:
        return [
            c
            for c in configs
            if max(
                c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"],
                c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_K"],
                c.kwargs["BLOCK_N"] * c.kwargs["BLOCK_K"],
            )
            <= 1048576
        ]
    if args["M"] == args["N"] == args["K"]:
        if args["A"].dtype == torch.float32:
            tile = (64, 128, 128) if args["M"] < 512 else (256, 128, 128)
        else:
            tile = (128, 128, 128) if args["M"] < 512 else (128, 256, 256)
        return [
            config
            for config in configs
            if tuple(config.kwargs[key] for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
            == tile
        ]
    # Large K tiles amortize loop overhead; short reductions need less padding.
    tiles = (
        {(128, 128, 128), (128, 256, 64)}
        if args["K"] <= 256
        else {(128, 128, 128), (128, 256, 256), (256, 128, 128)}
    )
    return [
        config
        for config in configs
        if tuple(config.kwargs[k] for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K")) in tiles
    ]


def prune_910c_addmm_stage3_tail(m, n, k, meta, stages):
    """Reject stage-3 BF16 ADDMM tiles with a narrow N and near-full M tail."""
    bm, bn, bk = (meta[key] for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
    return (
        stages == 3
        and 4 * n <= bn
        and 8 * (m % bm) >= 7 * bm
        and k >= max(4096, 16 * bk)
        and k % bk == 0
    )


class _AddMMFlagTuneTuner(ResourcePrunedGemmTuner):
    default_prune = staticmethod(_prune_addmm_configs)
    tile_prefix = "BLOCK"

    def _prune_candidates(self, configs, named_args, **kwargs):
        valid = super()._prune_candidates(configs, named_args, **kwargs)
        args = {**named_args, **kwargs}
        if args["A"].dtype == torch.bfloat16 and is_ascend_910c():
            m, n, k = (args[axis] for axis in ("M", "N", "K"))
            return [
                config
                for config in valid
                if not prune_910c_addmm_stage3_tail(
                    m, n, k, config.kwargs, config.num_stages
                )
            ]
        return valid


@libentry()
@libtuner(
    configs=_ADDMM_BASE_CONFIGS
    + [
        triton.Config(
            {"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k, "SPLIT_K": 1},
            num_warps=4,
            num_stages=2,
        )
        for m, n, k in (
            (64, 128, 128),
            (128, 256, 64),
            (128, 256, 256),
            (256, 128, 128),
        )
    ],
    key=[
        "M",
        "N",
        "K",
        "stride_am",
        "stride_ak",
        "stride_bk",
        "stride_bn",
        "stride_im",
        "stride_in",
        "stride_cm",
        "stride_cn",
        "BIAS_IS_VECTOR",
        "BIAS_IS_SCALAR",
        "BETA_IS_ZERO",
        "DOT_PAD_ONLY_K",
    ],
    warmup=5,
    rep=5,
    use_cuda_graph=True,
    flagtune_op_name="addmm",
    flagtune_expand_op_name="addmm_ascend",
    flagtune_yaml_path=_ADDMM_EXPAND_PATH,
    flagtune_op_id="flaggems/addmm",
    flagtune_variant="gemm",
    policy=_AddMMFlagTuneTuner,
)
@triton.heuristics(_ADDMM_HEURISTICS)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    A,
    B,
    bias,
    C,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
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
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    DOT_PAD_ONLY_K: tl.constexpr = False,
):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = grid_m * grid_n
    pid_z = tl.program_id(1)

    for pid in range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        # Visit neighboring M tiles before advancing N to improve B-tile reuse.
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        ram = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rbn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
        if DIVISIBLE_M:
            ram = tl.max_contiguous(tl.multiple_of(ram, BLOCK_M), BLOCK_M)
        if DIVISIBLE_N:
            rbn = tl.max_contiguous(tl.multiple_of(rbn, BLOCK_N), BLOCK_N)
        if EVEN_K:
            rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_K), BLOCK_K)
        A_ptr = A + ram[:, None] * stride_am + rk[None, :] * stride_ak
        B_ptr = B + rk[:, None] * stride_bk + rbn[None, :] * stride_bn
        if not DIVISIBLE_M:
            mask_m = ram < M
        if not DIVISIBLE_N:
            mask_n = rbn < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            if EVEN_K:
                if DIVISIBLE_M:
                    a = tl.load(A_ptr)
                else:
                    a = tl.load(A_ptr, mask=mask_m[:, None], other=0.0)
                if DIVISIBLE_N:
                    b = tl.load(B_ptr)
                else:
                    b = tl.load(B_ptr, mask=mask_n[None, :], other=0.0)
            else:
                k_remaining = K - k * (BLOCK_K * SPLIT_K)
                mask_k = rk < k_remaining
                a = tl.load(
                    A_ptr,
                    mask=(
                        mask_k[None, :]
                        if DIVISIBLE_M
                        else mask_m[:, None] & mask_k[None, :]
                    ),
                    other=0.0,
                )
                b = tl.load(
                    B_ptr,
                    mask=(
                        mask_k[:, None]
                        if DIVISIBLE_N
                        else mask_k[:, None] & mask_n[None, :]
                    ),
                    other=0.0,
                )
            if DOT_PAD_ONLY_K:
                extension.compile_hint(a, "dot_pad_only_k")
                extension.compile_hint(b, "dot_pad_only_k")
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            A_ptr += BLOCK_K * SPLIT_K * stride_ak
            B_ptr += BLOCK_K * SPLIT_K * stride_bk

        C_ptr = C + ram[:, None] * stride_cm + rbn[None, :] * stride_cn
        if DIVISIBLE_M and DIVISIBLE_N:
            mask = None
        elif DIVISIBLE_M:
            mask = mask_n[None, :]
        elif DIVISIBLE_N:
            mask = mask_m[:, None]
        else:
            mask = mask_m[:, None] & mask_n[None, :]
        acc = acc * alpha
        # A runtime branch here makes the large tiles exceed Ascend's UB.
        # Specialize only the zero case so NaN/Inf bias is never loaded then.
        if not BETA_IS_ZERO:
            if BIAS_IS_VECTOR:
                # Load a 1-D bias once per output-column tile.
                if DIVISIBLE_N:
                    bias_tile = tl.load(bias + stride_in * rbn)[None, :]
                else:
                    bias_tile = tl.load(
                        bias + stride_in * rbn,
                        mask=mask_n,
                        other=0.0,
                    )[None, :]
            elif BIAS_IS_SCALAR:
                bias_tile = tl.load(bias)
            else:
                bias_ptr = bias + stride_im * ram[:, None] + stride_in * rbn[None, :]
                if DIVISIBLE_M and DIVISIBLE_N:
                    bias_tile = tl.load(bias_ptr)
                else:
                    bias_tile = tl.load(bias_ptr, mask=mask, other=0.0)
            acc += bias_tile.to(acc.dtype) * beta
        if DIVISIBLE_M and DIVISIBLE_N:
            tl.store(C_ptr, acc.to(C.dtype.element_ty))
        else:
            tl.store(C_ptr, acc.to(C.dtype.element_ty), mask=mask)


def _launch_addmm_gemm(bias, mat1, mat2, out, alpha, beta):
    M, K = mat1.shape
    N = mat2.shape[1]
    # Align B row pitch when large NN GEMMs amortize the packing cost.
    # Padding is outside the logical N and is never loaded by the masked kernel.
    if (
        mat1.dtype == torch.bfloat16
        and mat1.stride(1) == 1
        and mat2.stride(1) == 1
        and mat2.stride(0) == N
        and out.stride(1) == 1
        and M >= 4096
        and N >= 128
        and K >= 512
        and N % 256 != 0
        and (N % 128 != 0 or K >= 4096)
    ):
        storage = torch.empty(
            (K, triton.cdiv(N, 256) * 256), device=mat2.device, dtype=mat2.dtype
        )
        packed = storage[:, :N]
        packed.copy_(mat2)
        mat2 = packed

    # Keep vector/scalar bias compact; broadcast strides cover other valid shapes.
    bias_is_vector = bias.ndim == 1 and bias.shape[0] == N
    bias_is_scalar = not bias_is_vector and bias.numel() == 1
    if bias_is_vector:
        bias_stride_m = 0
        bias_stride_n = bias.stride(0)
    elif bias_is_scalar:
        bias_stride_m = 0
        bias_stride_n = 0
    else:
        bias = bias.broadcast_to(out.shape)
        bias_stride_m = bias.stride(0)
        bias_stride_n = bias.stride(1)
    dot_out_dtype = tl.float32

    def _addmm_grid(META):
        grid_m = triton.cdiv(M, META["BLOCK_M"])
        grid_n = triton.cdiv(N, META["BLOCK_N"])
        grid_2d = grid_m * grid_n
        split_k = META.get("SPLIT_K", 1)
        # Bound workspace across all tuning candidates, accounting for SPLIT_K.
        capped_grid = min(grid_2d, _MAX_ADDMM_PROGRAMS // split_k)
        return (capped_grid, split_k)

    with torch_device_fn.device(mat1.device), _blockified():
        addmm_kernel[_addmm_grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias_stride_m,
            bias_stride_n,
            out.stride(0),
            out.stride(1),
            dot_out_dtype=dot_out_dtype,
            GROUP_M=8,
            enable_auto_blockify=True,
            BIAS_IS_VECTOR=bias_is_vector,
            BIAS_IS_SCALAR=bias_is_scalar,
            BETA_IS_ZERO=beta == 0,
            DOT_PAD_ONLY_K=(
                mat1.dtype in (torch.float16, torch.bfloat16)
                and mat1.stride(1) == 1
                and mat2.stride(1) == 1
                and out.stride(1) == 1
                and M >= 128
                and N >= 128
                and N % 16 == 0
                and K > 0
            ),
        )
    return out


@libentry()
@triton.jit
def _addmm_small_mm(A, B, P, N: tl.constexpr, BM: tl.constexpr):
    # Keep the matrix product separate from the Vector bias epilogue.
    BN: tl.constexpr = 128
    BK: tl.constexpr = 128
    pid = tl.program_id(0)
    row = pid // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    col = pid % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    ap = A + row[:, None] * N + kk[None, :]
    bp = B + kk[:, None] * N + col[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(N // BK):
        a = tl.load(ap, row[:, None] < N, other=0)
        b = tl.load(bp, col[None, :] < N, other=0)
        acc += tl.dot(a, b, allow_tf32=False)
        ap += BK
        bp += BK * N
    tl.store(
        P + row[:, None] * N + col[None, :],
        acc,
        (row[:, None] < N) & (col[None, :] < N),
    )


@libentry()
@triton.jit
def _addmm_small_add(P, Bias, C, N: tl.constexpr):
    total_elements = N * N
    total_blocks = tl.cdiv(total_elements, 8192)
    for block_id in range(tl.program_id(0), total_blocks, tl.num_programs(0)):
        idx = block_id * 8192 + tl.arange(0, 8192)
        value = tl.load(P + idx, idx < total_elements, other=0)
        bias = tl.load(Bias + idx, idx < total_elements, other=0).to(tl.float32)
        tl.store(C + idx, value + bias, idx < total_elements)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _addmm_small_finish(
    P,
    Bias,
    C,
    alpha,
    beta,
    N: tl.constexpr,
    UNIT: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
):
    total_elements = N * N
    total_blocks = tl.cdiv(total_elements, 8192)
    for block_id in range(tl.program_id(0), total_blocks, tl.num_programs(0)):
        idx = block_id * 8192 + tl.arange(0, 8192)
        value = tl.load(P + idx, idx < total_elements, other=0)
        result = value * alpha
        if not BETA_IS_ZERO:
            bias = tl.load(Bias + idx, idx < total_elements, other=0).to(tl.float32)
            if UNIT:
                result = value + bias
            else:
                result += bias * beta
        tl.store(C + idx, result, idx < total_elements)


def _launch_small_square_addmm(bias, mat1, mat2, out, alpha, beta):
    M, N = out.shape
    partial = torch.empty((M, N), dtype=torch.float32, device=mat1.device)
    bm = 64 if N < 512 else 256
    with torch_device_fn.device(mat1.device), _blockified():
        _addmm_small_mm[(triton.cdiv(M, bm) * triton.cdiv(N, 128),)](
            mat1,
            mat2,
            partial,
            N,
            bm,
            enable_auto_blockify=True,
            num_warps=4,
            num_stages=2,
        )
        grid_size = min(triton.cdiv(M * N, 8192), 65535)
        if alpha == 1 and beta == 1:
            _addmm_small_add[(grid_size,)](
                partial, bias, out, N, enable_auto_blockify=True
            )
        else:
            _addmm_small_finish[(grid_size,)](
                partial,
                bias,
                out,
                alpha,
                beta,
                N,
                UNIT=False,
                BETA_IS_ZERO=beta == 0,
                enable_auto_blockify=True,
            )
    return out


@triton.jit
def _addmm_scalar_dot(A, B, Bias, Out, K: tl.constexpr, BLOCK_K: tl.constexpr):
    k = tl.arange(0, BLOCK_K)
    a = tl.load(A + k, k < K, 0).to(tl.float32)
    b = tl.load(B + k, k < K, 0).to(tl.float32)
    result = tl.sum(a * b, axis=0) + tl.load(Bias).to(tl.float32)
    tl.store(Out, result)


@triton.jit
def _addmm_row_vector(
    A, B, Bias, Out, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr = 64
):
    k = tl.arange(0, K)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    a = tl.load(A + k).to(tl.float32)
    b = tl.load(B + k[:, None] * N + n[None, :]).to(tl.float32)
    value = tl.sum(a[:, None] * b, axis=0) + tl.load(Bias + n).to(tl.float32)
    tl.store(Out + n, value)


def _prune_addmm_row_vector(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    n, k = args["N"], args["K"]
    return [
        cfg
        for cfg in configs
        if n % cfg.kwargs["BN"] == 0
        and n // cfg.kwargs["BN"] <= 65535
        and k * cfg.kwargs["BN"] <= 16384
    ]


_addmm_row_vector_default = libentry()(
    libtuner(
        configs=[
            triton.Config(dict(BN=bn), num_warps=32, num_stages=2)
            for bn in (16, 32, 64, 128, 256)
        ],
        key=["N", "K"],
        prune_configs_by={"early_config_prune": _prune_addmm_row_vector},
        use_cuda_graph=True,
        rep=20,
        policy="flagtune",
        disable_aabs=True,
    )(_addmm_row_vector)
)


@triton.jit
def _addmm_wide_panel_kernel(
    A,
    Packed,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    panel = tile % (N // BN)
    m = (tile // (N // BN)) * BM + tl.arange(0, BM)
    n = tl.arange(0, BN)
    k0 = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(K // BK):
        k = step * BK + k0
        a = tl.load(A + m[:, None] * K + k[None, :], m[:, None] < M, 0)
        b = tl.load(Packed + panel * K * BN + k[:, None] * BN + n[None, :])
        extension.compile_hint(a, "dot_pad_only_k")
        extension.compile_hint(b, "dot_pad_only_k")
        acc = tl.dot(a, b, acc, allow_tf32=False)
    offset = m[:, None] * N + panel * BN + n[None, :]
    acc += tl.load(Bias + offset, m[:, None] < M, 0).to(tl.float32)
    tl.store(Out + offset, acc, m[:, None] < M)


addmm_wide_panel_kernel = libentry()(
    libtuner(
        configs=[
            triton.Config(dict(BM=bm, BK=256), num_warps=32, num_stages=2)
            for bm in (16, 128)
        ],
        key=["M", "N", "K"],
        prune_configs_by={"early_config_prune": prune_panel_gemm},
        use_cuda_graph=True,
        policy="flagtune",
        flagtune_op_name="addmm",
        flagtune_op_id="flaggems/addmm",
        flagtune_variant="wide_panels",
        flagtune_expand_op_name="panel_gemm_ascend",
        flagtune_yaml_path=_ADDMM_EXPAND_PATH,
        flagtune_dtype_resolver=gemm_dtype_resolver("addmm", rhs="Packed"),
        disable_aabs=True,
    )(_addmm_wide_panel_kernel)
)


def _launch_wide_addmm(a, b, bias, out):
    m, k = a.shape
    n = b.shape[1]
    bm = 16 if m <= 16 else 128
    bn = 256
    packed = torch.empty((n // bn, k, bn), device=b.device, dtype=b.dtype)
    with torch_device_fn.device(a.device), _blockified():
        _mm_pack_wide_panels_kernel[(n // bn * (k // 64),)](
            b, packed, n, k, bn, 64, multibuffer=False
        )
        launch_panel_gemm(
            _addmm_wide_panel_kernel,
            addmm_wide_panel_kernel,
            (a, packed, bias, out),
            (m, n, k),
            bm,
            256,
        )
    return out


def _is_contiguous_for_addmm_fallback(tensor):
    return tensor.is_contiguous() or (tensor.stride(0) > 1 and tensor.stride(1) > 1)


def _select_addmm_route(bias, mat1, mat2, *, alpha=1, beta=1, out=None):
    M, K = mat1.shape
    _, N = mat2.shape
    output_dtype = mat1.dtype if out is None else out.dtype
    if (
        alpha == beta == 1
        and bias.shape == (M, N)
        and bias.dtype == output_dtype == mat1.dtype
        and bias.is_contiguous()
        and (out is None or out.is_contiguous())
    ):
        if _use_nt_gemm(mat1, mat2):
            return "direct_nt", None
        if _use_narrow_gemv(mat1, mat2):
            return "narrow_gemv", None
        if _use_narrow_pack(mat1, mat2):
            return "packed_rhs", None
        if (
            mat1.dtype == mat2.dtype
            and mat1.dtype in (torch.float16, torch.bfloat16)
            and mat1.is_contiguous()
            and mat2.is_contiguous()
            and 0 < M <= 128
            and 131072 <= N <= 262144
            and N % 256 == 0
            and 1024 <= K <= 2048
            and K % 256 == 0
        ):
            return "wide_panels", None
        tile = _select_rectangular_tile(mat1, mat2)
        if (
            tile is None
            and mat1.dtype == mat2.dtype
            and mat1.dtype in (torch.float16, torch.bfloat16)
            and mat1.is_contiguous()
        ):
            if (
                M == 1
                and K == 128
                and 1024 <= N <= 8192
                and N % 512 == 0
                and mat2.transpose(0, 1).is_contiguous()
            ):
                tile = (16, 512, 128, 1, False)
            elif 0 < M <= 4 and N == 256 and K == 2048 and mat2.is_contiguous():
                tile = (16, 64, 256, 4, True)
        if tile is not None:
            return ("split_k" if tile[3] > 1 else "fixed_tile"), tile
        if (
            M == 1
            and K in (64, 128)
            and 1024 <= N <= 65535 * 64
            and N % 64 == 0
            and mat1.dtype == mat2.dtype
            and mat1.dtype in (torch.float16, torch.bfloat16)
            and mat1.is_contiguous()
            and mat2.is_contiguous()
        ):
            return "row_vector", None
        if (
            M == N == 1
            and 0 < K <= 2048
            and mat1.dtype == mat2.dtype
            and mat1.dtype in (torch.float16, torch.bfloat16)
            and mat1.is_contiguous()
            and mat2.is_contiguous()
        ):
            return "scalar_dot", None
    if (
        runtime.resolve_tuning_mode("addmm", supports_cost_model=True)
        == runtime.TuningMode.DEFAULT
        and M == N == K
        and (128 <= N < 512 or N == 1024)
        and N % 128 == 0
        and mat1.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and _is_contiguous_for_addmm_fallback(mat1)
        and _is_contiguous_for_addmm_fallback(mat2)
        and bias.shape == (M, N)
        and bias.is_contiguous()
        and (out is None or out.is_contiguous())
    ):
        return "small_square", None
    return "gemm", None


def _launch_addmm(bias, mat1, mat2, out, alpha, beta):
    M, K = mat1.shape
    N = mat2.shape[1]
    if M == 0 or N == 0:
        return out
    route, tile = _select_addmm_route(bias, mat1, mat2, alpha=alpha, beta=beta, out=out)
    if route == "direct_nt":
        return _launch_nt_gemm(mat1, mat2, out, bias, tuner=addmm_tiled_kernel)
    if route == "narrow_gemv":
        return _launch_narrow_gemv(
            mat1, mat2, out, bias, tuner=addmm_narrow_gemv_kernel
        )
    if route == "packed_rhs":
        return _launch_narrow_pack(mat1, mat2, out, bias)
    if route == "wide_panels":
        return _launch_wide_addmm(mat1, mat2, bias, out)
    if tile is not None:
        return _launch_tiled_gemm(
            mat1,
            mat2,
            out,
            tile,
            bias,
            tuner=addmm_tiled_kernel,
            split_tuner=addmm_split_kernel,
        )
    if route == "row_vector":
        with torch_device_fn.device(mat1.device):
            if mat1.dtype == torch.bfloat16:
                _addmm_row_vector_default[lambda meta: (N // meta["BN"],)](
                    mat1, mat2, bias, out, N, K
                )
            else:
                _addmm_row_vector[(N // 64,)](mat1, mat2, bias, out, N, K)
        return out
    if route == "scalar_dot":
        with torch_device_fn.device(mat1.device):
            _addmm_scalar_dot[(1,)](mat1, mat2, bias, out, K, triton.next_power_of_2(K))
        return out
    # Keep row- or column-contiguous views and materialize only general strides.
    if mat1.stride(0) > 1 and mat1.stride(1) > 1:
        mat1 = mat1.contiguous()
    if mat2.stride(0) > 1 and mat2.stride(1) > 1:
        mat2 = mat2.contiguous()

    if route == "small_square":
        return _launch_small_square_addmm(bias, mat1, mat2, out, alpha, beta)
    return _launch_addmm_gemm(bias, mat1, mat2, out, alpha, beta)


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_ASCEND ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M = mat1.shape[0]
    N = mat2.shape[1]
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    return _launch_addmm(bias, mat1, mat2, out, alpha, beta)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_ASCEND ADDMM_OUT")
    if out is not None and out.dtype != mat1.dtype:
        raise RuntimeError("out dtype must match mat1 dtype")
    return _addmm_out(bias, mat1, mat2, beta=beta, alpha=alpha, out=out)


def _addmm_out(bias, mat1, mat2, *, beta, alpha, out):
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M = mat1.shape[0]
    N = mat2.shape[1]
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        if out.device != mat1.device:
            raise RuntimeError("out must be on the same device as mat1")
        # Preserve aliased inputs before writing the output.
        if torch._C._is_alias_of(out, bias):
            bias = bias.clone()
        if torch._C._is_alias_of(out, mat1):
            mat1 = mat1.clone()
        if torch._C._is_alias_of(out, mat2):
            mat2 = mat2.clone()
        if out.shape != (M, N):
            out.resize_(M, N)
    return _launch_addmm(bias, mat1, mat2, out, alpha, beta)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS_ASCEND ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS_ASCEND ADDMM_DTYPE_OUT")
    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            f"mat1 and mat2 must have the same dtype, but got {mat1.dtype} and {mat2.dtype}"
        )
    if out.dtype != out_dtype:
        raise RuntimeError(
            "out_dtype must be the same as the dtype of the provided out tensor"
        )
    if not (
        out_dtype == mat1.dtype
        or (
            out_dtype == torch.float32 and mat1.dtype in (torch.float16, torch.bfloat16)
        )
    ):
        raise RuntimeError(
            "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs"
        )
    if bias.dtype != out_dtype and bias.dtype != mat1.dtype:
        raise RuntimeError("self dtype must match either out_dtype or mat1 dtype")

    bias_c = bias.to(out_dtype)
    return _addmm_out(bias_c, mat1, mat2, beta=beta, alpha=alpha, out=out)
