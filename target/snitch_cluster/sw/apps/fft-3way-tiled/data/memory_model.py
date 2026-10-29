#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `fft-3way-tiled` app. The partition-3 psum overlays
# the dead stage-1-4 scratch, while the assembled full H2 and the zero buffer survive, so:
#   peak = weights + full_H2 + zero + max(stages-1-4 scratch, partition-3 psum)
# Design: docs/dataflow/05_fft.md "fft-3way-tiled".

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (  # type: ignore[import]
    FP8,
    BF16,
    SEQ_LEN_UNROLL,
    D_INNER_UNROLL,
    TCDM_BYTES,
    align64,
    MemoryReport,
    run_model,
)


def _padded(x: int) -> int:
    # Lx_padded = Lx*dInnerUnroll/seqLenUnroll (per-16-tile pad), uniform for L1/L2/L3.
    assert (x * D_INNER_UNROLL) % SEQ_LEN_UNROLL == 0, f"L axis {x} must be even"
    return x * D_INNER_UNROLL // SEQ_LEN_UNROLL


def plan(params: dict) -> dict:
    """Buffer sizes + the zero-buffer size decision (shared with datagen)."""
    L = params["seqLen"]
    dModel = params["dModel"]
    L1, L2, L3 = params["L1"], params["L2"], params["L3"]
    nb_d = params["nb_tiles_A"]
    l3_tile = params.get("l3_tile", L3)

    assert L1 * L2 * L3 == L, f"L1*L2*L3 ({L1*L2*L3}) must equal seqLen ({L})"
    assert dModel % nb_d == 0, f"dModel ({dModel}) must be divisible by nb_tiles_A ({nb_d})"
    assert L3 % l3_tile == 0, f"L3 ({L3}) must be divisible by l3_tile ({l3_tile})"
    dM = dModel // nb_d
    L3t = l3_tile
    Lt = L1 * L2 * L3t
    L1p, L2p, L3p = _padded(L1), _padded(L2), _padded(L3)

    p = {
        "dM": dM,
        "nb_l3": L3 // l3_tile,
        "weight1": 2 * L1 * L1p * FP8 // 8,
        "weight2": 2 * L2 * 2 * L2p * FP8 // 8,
        "weight3": 2 * L3 * 2 * L3p * FP8 // 8,
        "in_tile": Lt * dM * FP8 // 8,
        "tw1_tile": 2 * Lt * FP8 // 8,
        "tw2_tile": 2 * L2 * L3t * FP8 // 8,
        "slot_size_tile": align64(2 * Lt * dM * BF16 // 8),  # gemm1/2 psum per l3-tile
        "full_h2": 2 * L * dM * FP8 // 8,
        "slot_size": align64(2 * L * dM * BF16 // 8),        # full partition-3 psum
    }
    p["hsize_tile"] = p["slot_size_tile"] // 2                # FP8 cmul1 scratch (H2)
    p["weights"] = align64(p["weight1"]) + align64(p["weight2"]) + align64(p["weight3"])
    p["scratch"] = (
        align64(p["in_tile"]) + align64(p["tw1_tile"]) + align64(p["tw2_tile"])
        + p["slot_size_tile"] + align64(p["hsize_tile"])
    )

    def peak(zero_bytes: int) -> int:
        return p["weights"] + align64(p["full_h2"]) + zero_bytes + max(p["scratch"], p["slot_size"])

    # The zero buffer feeds every accumulation's first psum read. Size it for the full
    # partition-3 psum when that fits, else for the per-tile gemm psums only.
    p["zero_bytes"] = p["slot_size"] if peak(p["slot_size"]) <= TCDM_BYTES else p["slot_size_tile"]
    p["peak"] = peak(p["zero_bytes"])
    return p


def build_report(params: dict) -> MemoryReport:
    p = plan(params)
    report = MemoryReport(
        "fft-3way-tiled",
        {
            "seqLen": params["seqLen"], "dModel": params["dModel"],
            "L1": params["L1"], "L2": params["L2"], "L3": params["L3"],
            "nb_tiles_A": params["nb_tiles_A"], "dM": p["dM"],
            "l3_tile": params.get("l3_tile", params["L3"]), "nb_l3": p["nb_l3"],
        },
    )
    report.add_section(
        "Weights (full, resident)",
        [("weight1", p["weight1"]), ("weight2", p["weight2"]), ("weight3", p["weight3"])],
    )
    report.add_section(
        "Stages 1-4 tile-local scratch",
        [
            ("in_tile (gathered)", p["in_tile"]),
            ("tw1_tile (gathered)", p["tw1_tile"]),
            ("tw2_tile (gathered)", p["tw2_tile"]),
            ("P_tile (gemm1/2 psum)", p["slot_size_tile"]),
            ("H2_tile (cmul1 out)", p["hsize_tile"]),
        ],
    )
    report.add_section(
        "Resident across phases",
        [("H2_full (partition-3 input)", p["full_h2"]), ("ZERO (K-step-0 psum source)", p["zero_bytes"])],
    )
    report.add_section("Partition 3 (overlays stages 1-4 scratch)", [("P3 (full BF16 psum)", p["slot_size"])])
    report.add_section(
        "L3 staging (not counted in TCDM)",
        [("partition3_out (L3)", 2 * params["seqLen"] * params["dModel"] * BF16 // 8)],
    )
    report.add_peak("TCDM peak", p["peak"])
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
