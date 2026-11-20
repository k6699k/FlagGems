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

import torch

from flag_gems.ops.baddbmm import BaddbmmFunction
from flag_gems.ops.baddbmm import baddbmm as generic_baddbmm
from flag_gems.ops.copy import copy_

from .bmm import _batch_product, _supported


def _forward(bias, A, B, beta, alpha):
    if alpha == 0 and beta in (0, 1):
        shape = (A.shape[0], A.shape[1], B.shape[2])
        expanded = bias.expand(shape)
        if beta == 0:
            return torch.zeros(shape, dtype=bias.dtype, device=bias.device)
        out = torch.empty(shape, dtype=bias.dtype, device=bias.device)
        return copy_(out, expanded)
    return _batch_product(A, B, bias, alpha, beta)


class _HopperBaddbmm(BaddbmmFunction):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        BaddbmmFunction._save_context(ctx, bias, A, B, beta, alpha)
        return _forward(bias, A, B, beta, alpha)


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    if bias.dtype != A.dtype or bias.device != A.device:
        raise RuntimeError(
            "baddbmm expects bias to have the same dtype and device as the matrices"
        )
    if not _supported(A, B) and not (alpha == 0 and beta in (0, 1)):
        return generic_baddbmm(bias, A, B, beta, alpha)
    if torch.is_grad_enabled() and (
        A.requires_grad or B.requires_grad or bias.requires_grad
    ):
        return _HopperBaddbmm.apply(bias, A, B, beta, alpha)
    return _forward(bias, A, B, beta, alpha)
