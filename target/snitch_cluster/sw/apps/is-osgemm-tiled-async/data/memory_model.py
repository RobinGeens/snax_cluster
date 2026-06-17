#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `is-osgemm-tiled-async` app.
#
# Parallel OSGEMM + ISGEMM with BOTH async rings live at once: the osCore A INPUT ring
# (refill) and the isCore BF16 PSUM OUTPUT ring (spill/reload). Layout (see src/main.c):
#   [ A_os ring(nb_slots) | B_os tile | D_os full | A_is ktile | B_is ktile | psum ring(nb_slots) ]
# D_os is full-resident (osCore writes tile slices); the full isCore psum lives in L3
# (not counted in TCDM). One dInnerUnroll invocation per K-step: nb_inv = dInner/dInnerUnroll.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (  # type: ignore[import]
    FP8, BF16, SEQ_LEN_UNROLL, D_INNER_UNROLL, MemoryReport, sequential_bytes, run_model,
)


def build_report(params: dict) -> MemoryReport:
    seqLen = params["seqLen"]
    dModel = params["dModel"]
    dInner = params["dInner"]
    nb_l_tiles = params["nb_l_tiles"]
    nb_slots = params["nb_slots"]
    Mu, Nu = SEQ_LEN_UNROLL, D_INNER_UNROLL

    report = MemoryReport("is-osgemm-tiled-async", {
        "seqLen": seqLen, "dModel": dModel, "dInner": dInner,
        "nb_l_tiles": nb_l_tiles, "nb_slots": nb_slots,
    })

    nb_inv = dInner // Nu                  # invocations = K-steps (both cores)
    L_tile = seqLen // nb_l_tiles

    # osCore (M3): A input ring, B tile, D full
    len_a_os_l_tile = L_tile * dModel * FP8 // 8
    len_b_os_tile = (dModel * dInner * FP8 // 8) // nb_inv
    len_d_os = seqLen * dInner * FP8 // 8

    # isCore (M4): A/B K-step tiles, BF16 psum ring
    len_a_is_ktile = (seqLen * dInner * FP8 // 8) // nb_inv
    len_b_is_ktile = (dInner * dModel * FP8 // 8) // nb_inv
    len_psum_l_tile = L_tile * dModel * BF16 // 8

    live_bufs = [
        (f"A_os ring ({nb_slots}x{len_a_os_l_tile}B)", nb_slots * len_a_os_l_tile),
        ("B_os tile", len_b_os_tile),
        ("D_os (full, resident)", len_d_os),
        ("A_is ktile", len_a_is_ktile),
        ("B_is ktile", len_b_is_ktile),
        (f"psum ring ({nb_slots}x{len_psum_l_tile}B)", nb_slots * len_psum_l_tile),
    ]
    report.add_section("Resident during compute (both rings live)", live_bufs)
    report.add_section("L3 staging (not counted in TCDM)",
                       [("full isCore psum (L3)", seqLen * dModel * BF16 // 8)])
    report.add_peak("IS+OSGEMM async (all buffers live)", sequential_bytes([s for _, s in live_bufs]))
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
