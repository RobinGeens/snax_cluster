#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `fft-3way-L8` app.
# Peak = weights + full_H2 + max(stages-1-4 scratch, partition-3 psum).
# See docs/dataflow/05_fft.md "fft-3way-L8".

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
    run_model,
)


def _padded(x: int) -> int:
    # Lx_padded = Lx*dInnerUnroll/seqLenUnroll (per-16-tile pad), uniform for L1/L2/L3.
    # Needs Lx even (L2,L3 multiples of 8, L1 of 16).
    assert (x * D_INNER_UNROLL) % SEQ_LEN_UNROLL == 0, f"L axis {x} must be even"
    return x * D_INNER_UNROLL // SEQ_LEN_UNROLL


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    L1, L2, L3 = params["L1"], params["L2"], params["L3"]
    nb_d = params["nb_tiles_A"]
    l3_tile = params.get("l3_tile", L3)

    assert L1 * L2 * L3 == L, f"L1*L2*L3 ({L1*L2*L3}) must equal seqLen ({L})"
    assert dModel % nb_d == 0, f"dModel ({dModel}) must be divisible by nb_tiles_A ({nb_d})"
    assert L3 % l3_tile == 0, f"L3 ({L3}) must be divisible by l3_tile ({l3_tile})"
    dM = dModel // nb_d
    # pad-m3: stages 1-2 run on a padded seqLenUnroll-tile when L3 < seqLenUnroll (see datagen).
    L3t = SEQ_LEN_UNROLL if L3 < SEQ_LEN_UNROLL else l3_tile
    Lt = L1 * L2 * L3t

    L1p, L2p, L3p = _padded(L1), _padded(L2), _padded(L3)

    report = MemoryReport(
        "fft-3way-L8",
        {
            "seqLen": L, "dModel": dModel, "L1": L1, "L2": L2, "L3": L3,
            "nb_tiles_A": nb_d, "dM": dM, "l3_tile": l3_tile, "nb_l3": L3 // l3_tile,
        },
    )

    # Weights: full, resident (depend on L only, broadcast over d).
    len_weight1 = 2 * L1 * L1p * FP8 // 8
    len_weight2 = 2 * L2 * 2 * L2p * FP8 // 8
    len_weight3 = 2 * L3 * 2 * L3p * FP8 // 8
    weights = align64(len_weight1) + align64(len_weight2) + align64(len_weight3)
    report.add_section(
        "Weights (full, resident)",
        [("weight1", len_weight1), ("weight2", len_weight2), ("weight3", len_weight3)],
    )

    # Stages 1-4 tile-local scratch (one l3-tile at a time).
    in_tile = Lt * dM * FP8 // 8
    tw1_tile = 2 * Lt * FP8 // 8
    tw2_tile = 2 * L2 * L3t * FP8 // 8
    slot_size_tile = align64(2 * Lt * dM * BF16 // 8)  # gemm1/2 psum per l3-tile
    hsize_tile = slot_size_tile // 2  # FP8 cmul/noop scratch (H1, H2)
    report.add_section(
        "Stages 1-4 tile-local scratch",
        [
            ("in_tile (gathered)", in_tile),
            ("tw1_tile (gathered)", tw1_tile),
            ("tw2_tile (gathered)", tw2_tile),
            ("P_tile (gemm1/2 psum)", slot_size_tile),
            ("H1_tile", hsize_tile),
            ("H2_tile (noop1/noop2)", hsize_tile),
        ],
    )

    # Assembled full H2 (partition-3 input): resident from stages 1-4 into partition 3.
    full_h2 = 2 * L * dM * FP8 // 8
    # Partition-3 psum: full BF16, overlays the dead stages-1-4 scratch (P3 = ptr_in).
    p3_full = align64(2 * L * dM * BF16 // 8)
    report.add_section("Resident across phases", [("H2_full (partition-3 input)", full_h2)])
    report.add_section("Partition 3 (overlays stages 1-4 scratch)", [("P3 (full BF16 psum)", p3_full)])

    report.add_section(
        "L3 staging (not counted in TCDM)",
        [("partition3_out (L3)", 2 * L * dModel * BF16 // 8)],
    )

    # Stages-1-4 scratch occupies [ptr_in .. ptr_H2_full); partition 3 overlays it with P3.
    stages14 = (
        align64(in_tile) + align64(tw1_tile) + align64(tw2_tile)
        + slot_size_tile + hsize_tile + align64(hsize_tile)
    )
    assert p3_full <= stages14, f"P3 ({p3_full}) must fit in the dead stages-1-4 scratch ({stages14})"
    peak = weights + align64(full_h2) + max(stages14, p3_full)
    report.add_peak("TCDM peak", peak)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
