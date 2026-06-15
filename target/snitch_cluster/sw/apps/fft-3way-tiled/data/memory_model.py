#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `fft-3way-tiled` app.
#
# The un-tiled 3-way kernel is run once per slice of dM = dModel / nb_tiles_A channels.
# Design: docs/dataflow/05_fft.md "fft-3way-tiled".

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (  # type: ignore[import]
    FP8,
    BF16,
    SEQ_LEN_UNROLL,
    D_INNER_UNROLL,
    align64,
    MemoryReport,
    sequential_bytes,
    run_model,
)


def _padded(x: int) -> int:
    # L*_padded: the seqLen-unroll tile count re-expressed in dInner-unroll units
    # (matches datagen's assert L1_padded // dInnerUnroll == L1 // seqLenUnroll).
    assert x % SEQ_LEN_UNROLL == 0, f"L axis {x} must be a multiple of {SEQ_LEN_UNROLL}"
    return (x // SEQ_LEN_UNROLL) * D_INNER_UNROLL


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    L1 = params["L1"]
    L2 = params["L2"]
    L3 = params["L3"]
    nb_d = params["nb_tiles_A"]

    assert L1 * L2 * L3 == L, f"L1*L2*L3 ({L1*L2*L3}) must equal seqLen ({L})"
    assert dModel % nb_d == 0, f"dModel ({dModel}) must be divisible by nb_tiles_A ({nb_d})"
    dM = dModel // nb_d

    L1p, L2p, L3p = _padded(L1), _padded(L2), _padded(L3)

    report = MemoryReport(
        "fft-3way-tiled",
        {
            "seqLen": L,
            "dModel": dModel,
            "L1": L1,
            "L2": L2,
            "L3": L3,
            "nb_tiles_A": nb_d,
            "dM": dM,
        },
    )

    # Always-live shared buffers (depend on L only, broadcast over d).
    len_weight1 = 2 * L1 * L1p * FP8 // 8
    len_weight2 = 2 * L2 * 2 * L2p * FP8 // 8
    len_weight3 = 2 * L3 * 2 * L3p * FP8 // 8
    len_twiddles1 = 2 * L * FP8 // 8
    len_twiddles2 = 2 * L2 * L3 * FP8 // 8
    report.add_section(
        "Always-live (shared, broadcast over d)",
        [
            ("weight1", len_weight1),
            ("weight2", len_weight2),
            ("weight3", len_weight3),
            ("twiddles1", len_twiddles1),
            ("twiddles2", len_twiddles2),
        ],
    )

    # Per-slice working buffers.
    len_in_slice = L * dM * FP8 // 8
    slot_size = align64(2 * L * dM * BF16 // 8)  # BF16 partition psum P
    hsize = slot_size // 2  # FP8 hadamard scratch (half a BF16 slot)
    report.add_section(
        "Per-slice working buffers",
        [
            ("in (FP8 input)", len_in_slice),
            ("P (BF16 partition psum)", slot_size),
            ("H1 (FP8 cmul out)", hsize),
            ("H2 (FP8 reorder out)", hsize),
        ],
    )

    # L3 staging (not counted in TCDM): the FULL-dModel assembled output.
    len_partition3_out = 2 * L * dModel * BF16 // 8
    report.add_section(
        "L3 staging (not counted in TCDM)",
        [
            ("partition3_out (L3)", len_partition3_out),
        ],
    )

    # Peak = shared + in laid out sequentially (each 64-aligned), then P, H1, H2
    # contiguous (matches the snrt_l1_next() bump pointers in src/fft.c).
    peak = (
        sequential_bytes([len_weight1, len_weight2, len_weight3, len_twiddles1, len_twiddles2, len_in_slice])
        + slot_size
        + hsize
        + hsize
    )
    report.add_peak("Per-slice resident", peak)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
