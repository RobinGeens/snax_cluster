#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `fft-tiled` app.
#
# Phase A (partition1 + hadamard + reorder): tiled along L2/dModel axis.
#   Always-live: weight1, weight2, twiddles
#   Ping-pong (×2): in_tile, partition1_out_tile, had_reord_tile
#
# Phase B (partition2): K-axis tiled.
#   Always-live: weight2, twiddles (weight1 not needed)
#   Single-buffered: had_reord_b_ktile, partition2_out (FULL)
#
# Phase B overlays Phase A's working region after barrier.
# L3 staging: hadamard_reordered (assembled in Phase A, read in Phase B).

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8, BF16, D_INNER_UNROLL, pad_to_unroll, align64,
    MemoryReport, run_model,
)


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    L1 = params["L1"]
    L2 = params["L2"]
    nb_A = params["nb_tiles"]
    nb_B = params.get("nb_tiles_B", nb_A)
    assert L1 * L2 == L
    L1_padded = pad_to_unroll(L1, D_INNER_UNROLL)
    L2_padded = pad_to_unroll(L2, D_INNER_UNROLL)

    report = MemoryReport("fft-tiled", {
        "seqLen": L, "dModel": dModel, "L1": L1, "L2": L2,
        "nb_tiles_A": nb_A, "nb_tiles_B": nb_B,
    })

    # Always-live (small weights + twiddles)
    len_weight1 = 2 * L1 * L1_padded * FP8 // 8
    len_weight2 = 2 * L2 * 2 * L2_padded * FP8 // 8
    len_twiddles = 2 * L * FP8 // 8

    always_live = [
        ("weight1",   len_weight1),
        ("weight2",   len_weight2),
        ("twiddles",  len_twiddles),
    ]
    report.add_section("Always-live (TCDM resident)", always_live)
    always_live_bytes = align64(len_weight1) + align64(len_weight2) + align64(len_twiddles)

    # Full tensor sizes
    len_in = L * dModel * FP8 // 8
    len_p1_out = 2 * L * dModel * BF16 // 8
    len_had_reord = 2 * L * dModel * FP8 // 8
    len_p2_out = 2 * L * dModel * BF16 // 8

    # Phase A: ping-pong (×2 each), tiled by nb_A
    phase_a_bufs = [
        ("in_tile",               len_in // nb_A),
        ("partition1_out_tile",   len_p1_out // nb_A),
        ("had_reord_a_tile",      len_had_reord // nb_A),
    ]
    report.add_section("Phase A ping-pong (×2 each, per tile)", phase_a_bufs)
    phase_a_pp = 2 * sum(align64(s) for _, s in phase_a_bufs)

    # Phase B: single-buffered (overlays Phase A region)
    phase_b_bufs = [
        ("had_reord_b_ktile",  len_had_reord // nb_B),
        ("partition2_out",     len_p2_out),  # FULL output
    ]
    report.add_section("Phase B single-buffered (overlays Phase A)", phase_b_bufs)
    phase_b = align64(len_had_reord // nb_B) + align64(len_p2_out)

    # L3 staging
    l3_bufs = [
        ("hadamard_reordered (L3)", len_had_reord),
    ]
    report.add_section("L3 staging (not counted in TCDM)", l3_bufs)

    peak_a = always_live_bytes + phase_a_pp
    peak_b = always_live_bytes + phase_b
    report.add_peak("Phase A (always-live + PP)", peak_a)
    report.add_peak("Phase B (always-live + single-buf)", peak_b)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
