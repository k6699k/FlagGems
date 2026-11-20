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

from .bmm import _batch_product, _supported


class _HopperBaddbmm(BaddbmmFunction):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        BaddbmmFunction._save_context(ctx, bias, A, B, beta, alpha)
        return _batch_product(A, B, bias, alpha, beta)


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    if not _supported(A, B) or bias.dtype != A.dtype or bias.device != A.device:
        return generic_baddbmm(bias, A, B, beta, alpha)
    if torch.is_grad_enabled() and (
        A.requires_grad or B.requires_grad or bias.requires_grad
    ):
        return _HopperBaddbmm.apply(bias, A, B, beta, alpha)
    return _batch_product(A, B, bias, alpha, beta)
