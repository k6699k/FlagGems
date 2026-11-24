# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import importlib

import pytest
import torch

import flag_gems
from flag_gems import runtime

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="requires Ascend NPU"
)


def test_mul_expand_spaces_have_distinct_kernel_contracts():
    mul = importlib.import_module(flag_gems.mul.__module__)
    scalar = runtime.ops_get_configs("mul", yaml_path=mul._MUL_EXPAND_PATH)
    tiled = runtime.ops_get_configs(
        "mul_broadcast_tiled", yaml_path=mul._MUL_EXPAND_PATH
    )
    assert len(scalar) == 18
    assert len(tiled) == 160

    def identity(config):
        return (
            tuple(sorted(config.kwargs.items())),
            config.num_warps,
            config.num_stages,
        )

    default_tiled = runtime.get_tuned_config("mul_broadcast_tiled")
    assert {identity(c) for c in default_tiled} <= {identity(c) for c in tiled}
    default_scalar = mul.mul_scalar_get_configs()
    assert len(default_scalar) == 18
    assert {identity(c) for c in default_scalar} == {identity(c) for c in scalar}
    assert {
        identity(c) for c in mul.mul_scalar_kernel.fn._flagtune_default_configs
    } == {identity(c) for c in default_scalar}
    assert all(set(config.kwargs) == {"BLOCK_SIZE"} for config in scalar)
    assert all(set(config.kwargs) == {"BLOCK_M", "BLOCK_N"} for config in tiled)
    assert (
        runtime.get_expand_config(
            "mul_broadcast_tiled", yaml_path=mul._MUL_EXPAND_PATH
        )["strategy"]
        == ["default"] * 7
    )


def test_tiled_pruning_respects_npu_program_limit():
    mul = importlib.import_module(flag_gems.mul.__module__)
    configs = runtime.ops_get_configs(
        "mul_broadcast_tiled", yaml_path=mul._MUL_EXPAND_PATH
    )
    args = {
        "M": 1048576,
        "N": 32,
        "dtype": "bfloat16",
        "a_s0": 1,
        "a_s1": 0,
        "b_s0": 0,
        "b_s1": 1,
    }
    selected = mul._prune_mul_tiled_configs(configs, args)
    assert selected
    assert all(
        ((args["M"] + c.kwargs["BLOCK_M"] - 1) // c.kwargs["BLOCK_M"])
        * ((args["N"] + c.kwargs["BLOCK_N"] - 1) // c.kwargs["BLOCK_N"])
        <= 65535
        for c in selected
    )
    assert all(c.kwargs["BLOCK_M"] >= 32 for c in selected)
    assert all(c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"] <= 1048576 for c in selected)


def test_tiled_pruning_keeps_legal_masked_and_small_tiles():
    import triton

    mul = importlib.import_module(flag_gems.mul.__module__)
    configs = [
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 4096}),
    ]
    args = {
        "M": 37,
        "N": 2048,
        "dtype": "bfloat16",
        "a_s0": 1,
        "a_s1": 0,
        "b_s0": 2048,
        "b_s1": 1,
    }
    assert mul._prune_mul_tiled_configs(configs, args) == configs


def test_scalar_pruning_only_removes_program_limit_violations():
    import triton

    mul = importlib.import_module(flag_gems.mul.__module__)
    configs = [
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 32768}),
    ]
    assert mul._prune_mul_flat_configs(configs, {"n_elements": 1}) == configs
    assert mul._prune_mul_flat_configs(configs, {"n_elements": 8388608}) == configs[1:]


def _track_graph_bench(monkeypatch, tuner):
    libentry = importlib.import_module("flag_gems.utils.libentry")
    monkeypatch.setattr(libentry.ConfigCache, "__contains__", lambda self, key: False)
    monkeypatch.setattr(libentry.BenchmarkCache, "get", lambda self, config: None)
    calls = []
    original = tuner.do_bench
    assert tuner._benchmark_protocol[0] == "ascend_graph"

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(tuner, "do_bench", tracked)
    return calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scalar_default_autotunes_without_flagtune(monkeypatch, dtype):
    monkeypatch.setenv("USE_FLAGTUNE", "0")
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "0")
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)
    mul = importlib.import_module(flag_gems.mul.__module__)
    tuner = mul._mul_persistent_default.fn
    calls = _track_graph_bench(monkeypatch, tuner)
    lhs = torch.randn((8192,), dtype=dtype, device="npu")
    actual = flag_gems.mul(lhs, 0.5)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, torch.mul(lhs, 0.5))
    expected_configs = mul._prune_persistent_mul(
        tuner._flagtune_default_configs, {"n_elements": lhs.numel()}
    )
    assert tuner._flagtune_mode is runtime.TuningMode.DEFAULT
    assert set(tuner.configs_timings) == set(expected_configs)
    assert tuner.benchmark_success_count == len(expected_configs)
    assert calls, "Default scalar tuning did not use the NPU graph benchmark"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mul_expanded_broadcast_candidates_use_npu_graph(monkeypatch, dtype):
    monkeypatch.setenv("FLAGTUNE_INCLUDE", "mul")
    monkeypatch.setenv("USE_FLAGTUNE_COST_MODEL", "0")
    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    mul = importlib.import_module(flag_gems.mul.__module__)
    tuner = mul.mul_broadcast_tiled_kernel.fn
    calls = _track_graph_bench(monkeypatch, tuner)
    prune = tuner.early_config_prune

    def sample_configs(configs, named_args, **kwargs):
        return prune(configs, named_args, **kwargs)[:2]

    monkeypatch.setattr(tuner, "early_config_prune", sample_configs)
    lhs = torch.randn((37, 1), dtype=dtype, device="npu")
    rhs = torch.randn((37, 2048), dtype=dtype, device="npu")
    actual = flag_gems.mul(lhs, rhs)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, torch.mul(lhs, rhs))
    assert tuner._flagtune_mode is runtime.TuningMode.EXPANDED
    assert calls, "expanded tuning did not execute the NPU graph benchmark"
