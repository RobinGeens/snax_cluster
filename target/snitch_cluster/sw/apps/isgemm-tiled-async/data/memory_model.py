#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `isgemm-tiled-async` app.
#
# ISGEMM with the BF16 psum held in an nb_slots-slot async ring (spilled/reloaded through
# L3 per L-tile, paced by R10). K (= dInner) is reduced over nb_k_tiles single-K-step
# invocations; A and B are double-buffered K-step tiles. Layout (see src/main.c):
#   [ A_k(2) | B_k(2) | psum ring (nb_slots slots) ]
# The full psum lives in L3 (not counted in TCDM). dim0=seqLen, dim1=dInner, dim2=dModel.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (  # type: ignore[import]
    FP8, BF16, SEQ_LEN_UNROLL, D_INNER_UNROLL, MemoryReport, sequential_bytes, run_model,
)


def build_report(params: dict) -> MemoryReport:
    seqLen = params["dim0"]
    dInner = params["dim1"]
    dModel = params["dim2"]
    nb_l_tiles = params["nb_l_tiles"]
    nb_slots = params["nb_slots"]
    Mu, Nu = SEQ_LEN_UNROLL, D_INNER_UNROLL

    report = MemoryReport("isgemm-tiled-async", {
        "dim0": seqLen, "dim1": dInner, "dim2": dModel,
        "nb_l_tiles": nb_l_tiles, "nb_slots": nb_slots,
    })

    nb_k_tiles = dInner // Nu              # SW-outer K reduction, one K-step per invocation
    L_tile = seqLen // nb_l_tiles
    psum_pos_per_l_tile = (L_tile // Mu) * dModel

    len_a_ktile = (seqLen * dInner * FP8 // 8) // nb_k_tiles
    len_b_ktile = (dInner * dModel * FP8 // 8) // nb_k_tiles
    len_psum_l_tile = psum_pos_per_l_tile * Mu * BF16 // 8

    live_bufs = [
        ("A_k (x2)", 2 * len_a_ktile),
        ("B_k (x2)", 2 * len_b_ktile),
        (f"psum ring ({nb_slots}x{len_psum_l_tile}B)", nb_slots * len_psum_l_tile),
    ]
    report.add_section("Resident during compute", live_bufs)
    report.add_section("L3 staging (not counted in TCDM)",
                       [("full psum (L3)", seqLen * dModel * BF16 // 8)])
    report.add_peak("ISGEMM async (all buffers live)", sequential_bytes([s for _, s in live_bufs]))
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
