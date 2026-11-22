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

import pytest
import torch
import triton

import flag_gems

from .accuracy_utils import SCALARS, gems_assert_close, to_reference
from .conftest import QUICK_MODE
from .gemm_utils import GEMM_DTYPES, accuracy_dtype_cases, accuracy_shapes

BMNK_SHAPES = accuracy_shapes("baddbmm", QUICK_MODE)
MNK_SHAPES = list(dict.fromkeys(shape[1:] for shape in BMNK_SHAPES))
BACKWARD_SHAPES = [(2, 16, 4096), (32, 64, 2048)]

GNK_SHAPES = [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]


FP8_MNK_SHAPES = [
    (128, 256, 512),
    (64, 128, 128),
    (256, 256, 256),
    (83, 7748, 3884),
    (84, 7168, 3884),
]


@pytest.mark.baddbmm
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend baddbmm")
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


@pytest.mark.baddbmm
@pytest.mark.parametrize(
    "dtype, scalar, batch, M, N, K, b_column_major",
    accuracy_dtype_cases("baddbmm", QUICK_MODE, scalars=SCALARS),
)
def test_baddbmm(batch, M, N, K, scalar, dtype, b_column_major):
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = (
        torch.randn((batch, N, K), dtype=dtype, device=flag_gems.device).transpose(1, 2)
        if b_column_major
        else torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    )
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)

    alpha = beta = scalar

    ref_out = torch.baddbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flag_gems.baddbmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.baddbmm_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_baddbmm_out(M, N, K, scalar, dtype):
    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((batch, M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)
    ref_out = to_reference(out, True)

    alpha = beta = scalar

    torch.baddbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta, out=ref_out)
    flag_gems.baddbmm_out(bias, mat1, mat2, alpha=alpha, beta=beta, out=out)

    gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.baddbmm
@pytest.mark.parametrize("M, N, K", BACKWARD_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_baddbmm_backward(M, N, K, scalar, dtype):
    batch = 2
    mat1 = torch.randn(
        (batch, M, K), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    mat2 = torch.randn(
        (batch, K, N), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    bias = torch.randn(
        (batch, M, N), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)
    alpha = beta = scalar

    ref_out = torch.baddbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flag_gems.baddbmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    _assert_baddbmm_gradients(
        res_out,
        ref_out,
        (bias, mat1, mat2),
        (ref_bias, ref_mat1, ref_mat2),
        bias_reduce_dim=K,
    )


@pytest.mark.baddbmm
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_baddbmm_backward_broadcast(dtype):
    batch, M, N, K = 2, 4, 16, 128
    alpha, beta = 0.25, 0.5
    inputs = tuple(
        torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
        for shape in ((N,), (batch, M, K), (batch, K, N))
    )
    ref_inputs = tuple(to_reference(x, True) for x in inputs)
    ref_out = torch.baddbmm(*ref_inputs, alpha=alpha, beta=beta)
    res_out = flag_gems.baddbmm(*inputs, alpha=alpha, beta=beta)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    _assert_baddbmm_gradients(
        res_out, ref_out, inputs, ref_inputs, bias_reduce_dim=batch * M
    )


def _assert_baddbmm_gradients(res_out, ref_out, inputs, ref_inputs, bias_reduce_dim):
    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    _, M, N = res_out.shape
    ref_in_bias, ref_in_grad1, ref_in_grad2 = torch.autograd.grad(
        ref_out, ref_inputs, ref_grad
    )
    res_in_bias, res_in_grad1, res_in_grad2 = torch.autograd.grad(
        res_out, inputs, out_grad
    )

    dtype = res_out.dtype
    gems_assert_close(res_in_bias, ref_in_bias, dtype, reduce_dim=bias_reduce_dim)
    gems_assert_close(res_in_grad1, ref_in_grad1, dtype, reduce_dim=N)
    gems_assert_close(res_in_grad2, ref_in_grad2, dtype, reduce_dim=M)
