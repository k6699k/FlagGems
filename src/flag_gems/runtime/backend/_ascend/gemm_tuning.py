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

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import torch
import triton

from flag_gems import runtime
from flag_gems.runtime.backend._ascend.blas_resources import (
    cube_tile_fits_910b,
    is_ascend_910b,
    prune_cube_tiles_910b,
)
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.libentry import LibTuner, LibTunerRunMode

FIXED_GEMM_EXPAND_PATH = str(Path(__file__).with_name("fixed_gemm_ascend_expand.yaml"))


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


def make_fixed_gemm_tuner(
    kernel,
    op,
    variant,
    key,
    preset,
    prune=None,
    *,
    space=None,
    stages=2,
    dtype_resolver=None,
):
    """Wrap an existing JIT kernel for opt-in tuning of a fixed fast path."""
    return libentry()(
        libtuner(
            configs=[triton.Config(preset, num_warps=4, num_stages=stages)],
            key=key,
            prune_configs_by={"early_config_prune": prune} if prune else None,
            use_cuda_graph=True,
            policy=AscendBlasTuner,
            flagtune_op_name=op,
            flagtune_op_id=f"flaggems/{op}",
            flagtune_variant=variant,
            flagtune_expand_op_name=f"{space or variant}_ascend",
            flagtune_yaml_path=FIXED_GEMM_EXPAND_PATH,
            flagtune_dtype_resolver=dtype_resolver,
            disable_aabs=True,
        )(kernel)
    )


def tune_fixed_gemm(tuner):
    # Keep the upstream preset until tuning is selected or requested offline.
    return (
        tuner.fn._run_mode is not LibTunerRunMode.NORMAL
        or runtime.flagtune_enabled(tuner.fn._flagtune_op_name)
        or runtime.resolve_cost_model_intent(supports_cost_model=True)
        is runtime.CostModelIntent.REQUIRED
    )


def gemm_dtype_resolver(op, rhs="B"):
    """Match the public input/output dtype order and omit scratch pointers."""
    names = ("A", rhs, "Out") if op in {"mm", "bmm"} else ("Bias", "A", rhs, "Out")

    def resolve(arguments):
        return tuple(arguments[name].dtype for name in names)

    return resolve


def is_ascend_910c():
    target = triton.runtime.driver.active.get_current_target()
    return str(target.arch).startswith("Ascend910_93")


def prune_910c_nt_column_tail(n, bn):
    """Reject wide masked column tiles for BF16 transposed RHS on 910C."""
    return bn >= 512 and n % bn != 0


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


def prune_910c_baddbmm_short_k(k, tile_k, stages):
    """Avoid 910C L0C conflicts when a pipelined K tile exceeds the reduction."""
    return stages >= 2 and tile_k > k


def prune_panel_gemm(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k, bn = (args[key] for key in ("M", "N", "K", "BN"))
    is_910b = is_ascend_910b()
    return [
        cfg
        for cfg in configs
        if k % cfg.kwargs["BK"] == 0
        and (
            cube_tile_fits_910b(
                cfg.kwargs["BM"],
                bn,
                cfg.kwargs["BK"],
                args["A"].dtype.itemsize,
                cfg.num_stages,
            )
            if is_910b
            else triton.cdiv(m, cfg.kwargs["BM"]) * triton.cdiv(n, bn) <= 65535
        )
    ]


def launch_panel_gemm(kernel, tuner, operands, shape, bm, bk):
    # BN is part of the packed layout and must agree with the pack kernel.
    m, n, _ = shape
    bn = 256
    panels = triton.cdiv(n, bn)
    options = dict(multibuffer=True, unit_flag=True, sync_solver=False)
    if tune_fixed_gemm(tuner):

        def grid(meta):
            return (min(triton.cdiv(m, meta["BM"]) * panels, 65535),)

        tuner[grid](*operands, *shape, BN=bn, **options)
    else:
        grid = (min(triton.cdiv(m, bm) * panels, 65535),)
        kernel[grid](*operands, *shape, bm, bn, bk, **options)


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
