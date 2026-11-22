# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="requires Ascend NPU"
)


@pytest.mark.parametrize(
    "op,variant,expected_count,inputs",
    [
        ("mv", "gemv", 216, {"M": 91, "N": 37}),
        ("addmm", "gemm", 132, {"M": 128, "N": 256, "K": 128}),
        ("baddbmm", "batched_gemm", 550, {"BATCH": 2, "M": 65, "N": 80, "K": 48}),
        ("addmm", "tiled_gemm", 86, {"M": 48, "N": 80, "K": 4096, "TRANSPOSED": 1}),
        ("bmm", "tiled_gemm", 86, {"BATCH": 2, "M": 64, "N": 96, "K": 4095}),
        ("baddbmm", "tiled_gemm", 86, {"BATCH": 2, "M": 64, "N": 96, "K": 4095}),
        ("addmm", "narrow_gemv", 20, {"M": 272, "K": 2048}),
        ("bmm", "narrow_gemv", 20, {"BATCH": 2, "M": 264, "K": 1024}),
        ("baddbmm", "narrow_gemv", 20, {"BATCH": 2, "M": 280, "K": 1024}),
        ("mm", "narrow_gemv", 22, {"M": 263, "K": 1009}),
        ("mm", "rectangular", 96, {"M": 16, "N": 64, "K": 2048}),
        ("mm", "packed_narrow", 28, {"M": 128, "N": 16, "K": 2048}),
        ("mm", "wide_panels", 24, {"M": 8, "N": 131072, "K": 1280}),
        ("mm", "large_panels", 24, {"M": 1088, "N": 8449, "K": 2304}),
        ("addmm", "wide_panels", 24, {"M": 8, "N": 131072, "K": 1024}),
    ],
)
def test_ascend_blas_contract_matches_runtime(op, variant, expected_count, inputs):
    from triton.flagtune.training.ranker import load_training_variant

    from flag_gems.flagtune.offline.contracts.operator import (
        load_operator_benchmark_spec,
    )
    from flag_gems.flagtune.offline.train.config_space import (
        runtime_configs_for_variant,
    )
    from flag_gems.utils.libentry import LibTuner, find_flagtune_benchmark_target

    public = getattr(flag_gems, op)
    assert "_ascend.ops." in public.__module__
    _, tuner = find_flagtune_benchmark_target(public, f"flaggems/{op}", variant)
    assert isinstance(tuner, LibTuner.get("flagtune"))
    assert tuner._flagtune_op_id == "flaggems/" + op
    assert tuner._flagtune_variant == variant
    contract = (
        Path(flag_gems.__file__).parent
        / "flagtune/offline/contracts/configs"
        / (op + "_ascend_flagtune_configs.yaml")
    )
    assert load_operator_benchmark_spec(contract).op_id == "flaggems/" + op
    model_variant = load_training_variant(contract, variant)
    configs = runtime_configs_for_variant("flaggems/" + op, variant, platform="ascend")
    assert len(configs) == expected_count
    config_dicts = [
        dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages) for c in configs
    ]
    assert all(model_variant.param_space.validate(c) for c in config_dicts)
    assert model_variant.build_feature_matrix(inputs, config_dicts).shape == (
        len(configs),
        len(model_variant.feature_names),
    )


@pytest.mark.parametrize("reduction", [2048, 2049, 4097])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ascend_legacy_mv_prunes_by_reduction_stride(reduction, dtype):
    prune = flag_gems.mv.__globals__["_prune_mv_configs"]
    safe = triton.Config({"BLOCK_M": 64, "BLOCK_N": 32})
    large = triton.Config({"BLOCK_M": 512, "BLOCK_N": 16})
    args = {"M": reduction, "N": 129, "A": SimpleNamespace(dtype=dtype)}
    assert prune([safe, large], {**args, "stride_am": 129}) == [safe]
    assert prune([safe, large], {**args, "stride_am": 1}) == [safe, large]


@pytest.mark.parametrize(
    "n,k,bn,bk",
    [
        (128, 64, 256, 128),
        (129, 80, 256, 128),
        (384, 96, 256, 128),
        (192, 64, 128, 512),
    ],
)
def test_ascend_tiled_gemm_prunes_wide_nt_bias_with_k_padding(
    n, k, bn, bk, monkeypatch
):
    symbols = flag_gems.bmm.__globals__
    monkeypatch.setitem(symbols, "is_ascend_910b", lambda: False)
    monkeypatch.setitem(symbols, "is_ascend_910c", lambda: True)
    prune = symbols["_prune_tiled_gemm_configs"]
    safe = triton.Config({"BM": 32, "BN": 128, "BK": 128}, num_stages=1)
    wide = triton.Config({"BM": 32, "BN": bn, "BK": bk}, num_stages=1)
    args = {
        "M": 128,
        "N": n,
        "K": k,
        "BATCH": 2,
        "A": SimpleNamespace(dtype=torch.bfloat16),
        "TRANSPOSED": True,
        "WITH_BIAS": True,
        "HINT": False,
    }
    assert prune([safe, wide], args) == [safe]
    for safe_layout in (
        {"WITH_BIAS": False},
        {"TRANSPOSED": False},
        {"K": bk},
        {"N": bn},
    ):
        assert prune([safe, wide], {**args, **safe_layout}) == [safe, wide]


def test_ascend_bmm_default_prunes_by_tile_constraints():
    prune = flag_gems.bmm.__globals__["_prune_bmm_configs"]
    tile = {"TILE_M": 32, "TILE_N": 64, "TILE_K": 128, "GROUP_M": 1}
    safe = triton.Config(tile)
    invalid = triton.Config({**tile, "TILE_M": 48})
    for n in (248320, 248321):
        args = {"BATCH": 4, "M": 2048, "N": n, "K": 2048}
        assert prune([safe, invalid], args) == [safe]


def test_ascend_910b_cube_budget():
    from flag_gems.runtime.backend._ascend.ops.bmm import cube_tile_fits_910b

    assert cube_tile_fits_910b(128, 256, 256, 2, 2)
    assert not cube_tile_fits_910b(128, 256, 256, 2, 3)
    assert not cube_tile_fits_910b(128, 256, 256, 4, 2)
    assert not cube_tile_fits_910b(256, 256, 32, 2, 1)


def test_ascend_910b_tiled_gemm_applies_resource_budget(monkeypatch):
    symbols = flag_gems.bmm.__globals__
    monkeypatch.setitem(symbols, "is_ascend_910b", lambda: True)
    monkeypatch.setitem(symbols, "is_ascend_910c", lambda: False)
    prune = symbols["_prune_tiled_gemm_configs"]
    stage2 = triton.Config({"BM": 128, "BN": 256, "BK": 256}, num_stages=2)
    stage3 = triton.Config(stage2.kwargs, num_stages=3)
    args = {
        "M": 129,
        "N": 384,
        "K": 96,
        "BATCH": 2,
        "A": SimpleNamespace(dtype=torch.bfloat16),
        "TRANSPOSED": True,
        "WITH_BIAS": True,
        "HINT": False,
    }
    assert prune([stage2, stage3], args) == []
    assert prune([stage2, stage3], {**args, "WITH_BIAS": False}) == [stage2]
    assert prune([stage2, stage3], {**args, "N": 256}) == [stage2]


def test_ascend_910b_direct_nt_default_covers_narrow_output(monkeypatch):
    from flag_gems import runtime

    symbols = flag_gems.bmm.__globals__
    monkeypatch.setitem(symbols, "is_ascend_910b", lambda: True)
    monkeypatch.setitem(symbols, "is_ascend_910c", lambda: False)
    tuner = symbols["bmm_direct_nt_kernel"].fn
    configs, _ = tuner._flagtune_configs_for_mode("bmm", runtime.TuningMode.DEFAULT)
    args = {
        "M": 65,
        "N": 16,
        "K": 128,
        "BATCH": 2,
        "A": SimpleNamespace(dtype=torch.bfloat16),
        "TRANSPOSED": True,
        "WITH_BIAS": False,
        "HINT": False,
    }
    assert [c.kwargs["BN"] for c in tuner.early_config_prune(configs, args)] == [16]


def test_ascend_910b_mv_ub_and_grid(monkeypatch):
    symbols = flag_gems.mv.__globals__
    monkeypatch.setitem(symbols, "is_ascend_910b", lambda: True)
    prune = symbols["_prune_mv_configs"]
    small = triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_stages=1)
    oversized = triton.Config({"BLOCK_M": 4096, "BLOCK_N": 32}, num_stages=1)
    args = {
        "A": SimpleNamespace(dtype=torch.float32),
        "N": symbols["_MV_910B_MAX_ROWS"],
        "stride_am": 2,
    }
    assert prune([small, oversized], args) == [small]
    assert prune([small, oversized], {**args, "N": args["N"] + 1}) == []


def test_ascend_910b_mv_empty_shape():
    if not flag_gems.mv.__globals__["is_ascend_910b"]():
        pytest.skip("requires Ascend 910B")
    for n, k in [(0, 7), (11, 0)]:
        inp = torch.empty((n, k), device="npu", dtype=torch.float32)
        vec = torch.empty((k,), device=inp.device, dtype=inp.dtype)
        torch.testing.assert_close(flag_gems.mv(inp, vec).cpu(), torch.zeros(n))
