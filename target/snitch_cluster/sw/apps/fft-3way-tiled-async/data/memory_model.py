#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for `fft-3way-tiled-async`.
# Peak = weights + max(stages-1-4 scratch, partition-3 buffers); see docs/dataflow/05_fft.md.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (  # type: ignore[import]
    FP8, BF16, SEQ_LEN_UNROLL, D_INNER_UNROLL, align64,
    MemoryReport, run_model,
)


def _padded(x: int) -> int:
    assert x % SEQ_LEN_UNROLL == 0, f"L axis {x} must be a multiple of {SEQ_LEN_UNROLL}"
    return (x // SEQ_LEN_UNROLL) * D_INNER_UNROLL


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    L1, L2, L3 = params["L1"], params["L2"], params["L3"]
    nb_d = params["nb_tiles_A"]
    l3_tile = params.get("l3_tile", L3)
    nb_ntile = params.get("nb_ntile", 1)

    assert L1 * L2 * L3 == L
    assert dModel % nb_d == 0
    assert L3 % l3_tile == 0
    dM = dModel // nb_d
    L3t = l3_tile
    Lt = L1 * L2 * L3t
    N_3 = dM * L1 * L2
    assert N_3 % nb_ntile == 0

    L1p, L2p, L3p = _padded(L1), _padded(L2), _padded(L3)

    report = MemoryReport("fft-3way-tiled-async", {
        "seqLen": L, "dModel": dModel, "L1": L1, "L2": L2, "L3": L3,
        "nb_tiles_A": nb_d, "dM": dM, "l3_tile": l3_tile, "nb_ntile": nb_ntile,
    })

    len_weight1 = 2 * L1 * L1p * FP8 // 8
    len_weight2 = 2 * L2 * 2 * L2p * FP8 // 8
    len_weight3 = 2 * L3 * 2 * L3p * FP8 // 8
    weights = align64(len_weight1) + align64(len_weight2) + align64(len_weight3)
    report.add_section("Weights (full, resident)", [
        ("weight1", len_weight1), ("weight2", len_weight2), ("weight3", len_weight3),
    ])

    in_tile = Lt * dM * FP8 // 8
    tw1_tile = 2 * Lt * FP8 // 8
    tw2_tile = 2 * L2 * L3t * FP8 // 8
    slot_size_tile = align64(2 * Lt * dM * BF16 // 8)   # gemm1/2 psum per l3-tile
    hsize_tile = slot_size_tile // 2                     # FP8 cmul scratch (H2, x2 ping-pong)
    full_h2 = 2 * L * dM * FP8 // 8                       # partition-3 input (staged to L3)
    h2_ntile = full_h2 // nb_ntile                       # one gathered N-tile of H2 (TCDM)
    p3_ntile = align64(2 * L * dM * BF16 // 8) // nb_ntile
    report.add_section("TCDM stages 1-4 (resident)", [
        ("in_tile (gathered)", in_tile),
        ("tw1_tile (gathered)", tw1_tile),
        ("tw2_tile (gathered)", tw2_tile),
        ("P_tile (gemm1/2 psum)", slot_size_tile),
        ("H2_tile (cmul out, x2)", 2 * hsize_tile),
    ])
    # Overlay the dead stages-1-4 scratch (ptr_h2ntile = ptr_in); not added to peak.
    report.add_section("TCDM partition 3 (overlays stages 1-4)", [
        ("h2_ntile (gemm3 gather)", h2_ntile),
        ("P3 (one N-tile psum)", p3_ntile),
    ])

    report.add_section("L3 staging (not counted in TCDM)", [
        ("partition3_out (L3)", 2 * L * dModel * BF16 // 8),
        ("full H2 (L3)", full_h2),
    ])

    stages14 = (
        align64(in_tile) + align64(tw1_tile) + align64(tw2_tile)
        + slot_size_tile + 2 * hsize_tile
    )
    partition3 = align64(h2_ntile) + p3_ntile
    peak = weights + max(stages14, partition3)
    report.add_peak("TCDM peak", peak)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
