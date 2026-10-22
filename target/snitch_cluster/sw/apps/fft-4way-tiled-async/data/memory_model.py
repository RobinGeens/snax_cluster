#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for `fft-4way-tiled-async`. The stage-3/4 full buffers overlay
# the dead stage-1-2 tile scratch, so:
#   peak = weights + max(stages-1-2 tile scratch, stage-3/4 full scratch)
# Design: docs/dataflow/05_fft.md "fft-4way-tiled-async".

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (  # type: ignore[import]
    FP8, BF16, SEQ_LEN_UNROLL, D_INNER_UNROLL, align64,
    MemoryReport, run_model,
)


def _padded(x: int) -> int:
    # Generator pads each DFT axis by D_INNER_UNROLL/SEQ_LEN_UNROLL (e.g. 16->24, 8->12).
    return x * D_INNER_UNROLL // SEQ_LEN_UNROLL


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    L1, L2, L3, L4 = params["L1"], params["L2"], params["L3"], params["L4"]
    nb_d = params["nb_tiles_A"]
    l3_tile = params.get("l3_tile", L3)
    nb_m4 = params.get("nb_m4", 1)
    nb_ntile = params.get("nb_ntile", 1)
    dM = dModel // nb_d
    L3t = l3_tile
    L4t = L4 // nb_m4
    Lt = L1 * L2 * L3t * L4t

    L1p, L2p, L3p, L4p = _padded(L1), _padded(L2), _padded(L3), _padded(L4)

    report = MemoryReport("fft-4way-tiled-async", {
        "seqLen": L, "dModel": dModel, "L1": L1, "L2": L2, "L3": L3, "L4": L4,
        "nb_tiles_A": nb_d, "dM": dM, "l3_tile": l3_tile, "nb_m4": nb_m4, "nb_ntile": nb_ntile,
    })

    len_w1 = 2 * L1 * L1p * FP8 // 8
    len_w2 = 2 * L2 * 2 * L2p * FP8 // 8
    len_w3 = 2 * L3 * 2 * L3p * FP8 // 8
    len_w4 = 2 * L4 * 2 * L4p * FP8 // 8
    weights = align64(len_w1) + align64(len_w2) + align64(len_w3) + align64(len_w4)
    report.add_section("Weights (full, resident)", [
        ("weight1", len_w1), ("weight2", len_w2), ("weight3", len_w3), ("weight4", len_w4),
    ])

    # packed3 (assembled partition3 input) is resident across both phases.
    packed3 = align64(2 * L * dM * FP8 // 8)

    # Stages 1-2: per-tile local buffers (m3-blocks via l3_tile or m4-blocks via nb_m4).
    in_tile = Lt * dM * FP8 // 8
    tw1_tile = 2 * L1 * L2 * L3t * L4t * FP8 // 8
    tw2_tile = 2 * L2 * L3t * L4t * FP8 // 8
    slot_tile = align64(2 * Lt * dM * BF16 // 8)   # gemm1/2 psum per tile
    hsize_tile = slot_tile // 2                     # FP8 cmul/noop scratch (H1, H2)
    report.add_section("TCDM stages 1-2 (per tile + resident packed3)", [
        ("packed3 (resident)", packed3),
        ("in_tile", in_tile), ("tw1_tile", tw1_tile), ("tw2_tile", tw2_tile),
        ("P_tile (gemm1/2 psum)", slot_tile), ("H1_tile", hsize_tile), ("H2_tile", hsize_tile),
    ])

    # Stages 3-4: per (k1,k2) N-chunk. These overlay the dead stages-1-2 scratch.
    P3 = align64(2 * L * dM * BF16 // (8 * nb_ntile))
    H3 = align64(2 * L * dM * FP8 // (8 * nb_ntile))
    packed4 = align64(2 * L * dM * FP8 // (8 * nb_ntile))
    P4 = align64(2 * L * dM * BF16 // (8 * nb_ntile))
    report.add_section("TCDM stages 3-4 (per N-chunk, overlay stages 1-2)", [
        ("packed3 (resident)", packed3), ("P3", P3), ("H3", H3),
        ("packed4 (gemm4 input)", packed4), ("P4", P4),
    ])

    report.add_section("L3 staging (not counted in TCDM)", [
        ("partition4_out (L3)", 2 * L * dModel * BF16 // 8),
    ])

    stages12 = (
        align64(in_tile) + align64(tw1_tile) + align64(tw2_tile)
        + slot_tile + hsize_tile + hsize_tile
    )
    stages34 = P3 + H3 + packed4 + P4
    peak = weights + packed3 + max(stages12, stages34)
    report.add_peak("TCDM peak", peak)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
