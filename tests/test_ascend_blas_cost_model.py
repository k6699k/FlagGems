# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import pytest
import torch
import triton

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="requires Ascend NPU"
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


@pytest.mark.baddbmm
@pytest.mark.parametrize("n, expected_bk", [(80, 128), (128, 256)])
def test_baddbmm_ascend_910b_nt_bias_tail(monkeypatch, n, expected_bk):
    from contextlib import nullcontext

    symbols = flag_gems.bmm.__globals__
    monkeypatch.setenv("USE_FLAGTUNE", "0")
    monkeypatch.setitem(symbols, "is_ascend_910b", lambda: True)
    monkeypatch.setitem(symbols, "is_ascend_910c", lambda: False)
    monkeypatch.setattr(symbols["torch_device_fn"], "device", lambda *_: nullcontext())
    tiles = []

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                tiles.append(args[-5:-2])

            return launch

    monkeypatch.setitem(symbols, "_tiled_gemm_kernel", Capture())
    options = {"device": "meta", "dtype": torch.bfloat16}
    a = torch.empty((2, 128, 256), **options)
    b = torch.empty((2, n, 256), **options).transpose(-2, -1)
    bias = torch.empty((2, 128, n), **options)
    assert flag_gems.baddbmm(bias, a, b).shape == bias.shape
    assert tiles == [(64, 64, expected_bk)]

    candidates = [
        triton.Config({"BM": 64, "BN": 64, "BK": bk}, num_stages=2) for bk in (128, 256)
    ]
    args = {
        "A": a,
        "BATCH": 2,
        "M": 128,
        "N": n,
        "K": 256,
        "TRANSPOSED": True,
        "WITH_BIAS": True,
        "HINT": False,
    }
    prune = symbols["_prune_tiled_gemm_configs"]
    assert prune(candidates, args) == [
        config for config in candidates if config.kwargs["BK"] <= expected_bk
    ]
    assert prune(candidates, {**args, "WITH_BIAS": False}) == candidates
