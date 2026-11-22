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

"""Operator-agnostic route resolver dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from .common import backend_module, platform
from .mm import resolve_mm_route

RouteResolver = Callable[[Any, Any, Mapping[str, Any] | None], dict[str, Any]]
ROUTE_RESOLVERS: dict[str, RouteResolver] = {
    "mm": resolve_mm_route,
    "flaggems/mm": resolve_mm_route,
}


_BLAS_OPS = frozenset({"addmm", "bmm", "baddbmm"})


def _resolve_ascend_blas_route(op, a, b, bias, context):
    vendor = platform(context, a)
    if vendor != "ascend":
        raise ValueError(f"no {op} route resolver for platform {vendor!r}")
    module = backend_module(op, vendor)
    select_route = getattr(module, f"_select_{op}_route")
    route, tile = select_route(a, b) if op == "bmm" else select_route(bias, a, b)
    variant = "gemm" if op == "addmm" else "batched_gemm"
    if route in {"direct_nt", "fixed_tile"}:
        variant = "tiled_gemm"
    elif route in {"narrow_gemv", "wide_panels", "direct_nt_tiled", "column_gemv"}:
        variant = route
    adapted = route in {variant, "direct_nt", "fixed_tile"}
    stage = "compute" if route == "wide_panels" else "public"
    return {
        "platform": vendor,
        "physical_route": route,
        "route_variant": route,
        "cost_model_variant": variant if adapted else None,
        "tuning_variant": variant if adapted else None,
        "stage": stage if adapted else None,
        "latency_scope": f"{stage}_kernel",
        "dynamic_inputs": {"tile": list(tile)} if tile is not None else {},
        "adapted": adapted,
        "reason": "adapted variant" if adapted else f"fixed-config route: {route}",
    }


def _resolve_ascend_mv_route(a, b, context):
    module = backend_module("mv", "ascend")
    requested = context.get("requested_tuning_variant")
    column = requested != "gemv" and module._can_launch_column_mv(a, b)
    variant = "column_gemv" if column else "gemv"
    stage = "compute" if column else "public"
    return {
        "platform": "ascend",
        "physical_route": variant,
        "route_variant": variant,
        "cost_model_variant": variant,
        "tuning_variant": variant,
        "stage": stage,
        "latency_scope": f"{stage}_kernel",
        "dynamic_inputs": {},
        "adapted": True,
        "reason": "adapted variant",
    }


def has_route_resolver(op_id: str) -> bool:
    """Whether real tensors, rather than contract predicates, select a route."""
    key = str(op_id)
    name = key.rsplit("/", 1)[-1]
    if name in _BLAS_OPS or name == "mv":
        from flag_gems import runtime

        return runtime.device.vendor_name == "ascend"
    return key in ROUTE_RESOLVERS or name in ROUTE_RESOLVERS


def tuning_variant(spec: Any, variant: str) -> str:
    """Use the contract's stage binding without interpreting kernel names."""
    info = spec.operator_info.variants.get(variant)
    return str(getattr(info, "route_binding", None) or variant)


def register_route_resolver(op_id: str, resolver: RouteResolver) -> None:
    """Register an operator resolver without changing shared dispatch code."""
    ROUTE_RESOLVERS[str(op_id)] = resolver


def resolve_route(
    op_id: str,
    a: Any,
    b: Any = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a route through the resolver registered for ``op_id``.

    The compatibility ``a``/``b`` form remains supported for existing MM
    callers.  New executors may pass a tensor mapping; conventional ``a``/``b``
    and ``lhs``/``rhs`` names are accepted without making operator resolvers
    depend on a particular benchmark YAML.
    """
    key = str(op_id)
    name = key.rsplit("/", 1)[-1]
    input_a = a.get("a") if isinstance(a, Mapping) else a
    if name == "mv" and platform(runtime_context, input_a) == "ascend":
        if isinstance(a, Mapping) and b is None:
            a, b = a["a"], a["b"]
        return _resolve_ascend_mv_route(a, b, runtime_context or {})
    if name in _BLAS_OPS and platform(runtime_context, input_a) == "ascend":
        bias = None
        if isinstance(a, Mapping) and b is None:
            a, b, bias = a["a"], a["b"], a.get("bias")
        return _resolve_ascend_blas_route(name, a, b, bias, runtime_context or {})
    try:
        resolver = ROUTE_RESOLVERS.get(key) or ROUTE_RESOLVERS[key.rsplit("/", 1)[-1]]
    except KeyError as exc:
        raise ValueError(
            f"no route resolver registered for operator {op_id!r}"
        ) from exc
    if isinstance(a, Mapping) and b is None:
        tensors = a
        a = tensors.get("a", tensors.get("lhs"))
        b = tensors.get("b", tensors.get("rhs"))
    return resolver(a, b, runtime_context)


__all__ = [
    "ROUTE_RESOLVERS",
    "has_route_resolver",
    "register_route_resolver",
    "resolve_route",
    "tuning_variant",
]
