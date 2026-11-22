# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import triton

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


def masked_bias_rhs_fits_910b(n, k, bn, bk, element_bytes):
    """Leave room for padding when a transposed BF16 RHS has two masked axes."""
    # A full L0B-sized tile with this layout stalled the bias kernel on 910B4-1.
    return n % bn == 0 or k >= bk or bn * bk * element_bytes < _L0B_BYTES


def vector_tile_fits_910b(rows, reduction, element_bytes, stages, *, strided=False):
    """Budget FP32 working tiles and staged inputs in Unified Buffer."""
    if strided:
        # The legacy gather lowering needs additional UB scratch. Keep its
        # empirically validated accumulator bound for strided reductions.
        if max(rows, 2) * reduction * _FP32_BYTES >= _UB_BYTES // 12:
            return False
    staged_bytes = max(stages, 1) * element_bytes
    matrix_bytes = rows * reduction * (2 * _FP32_BYTES + staged_bytes)
    vector_bytes = reduction * (_FP32_BYTES + staged_bytes) + rows * _FP32_BYTES
    return matrix_bytes + vector_bytes <= _UB_BYTES
