#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `osgemm-tiled-async` app.
#
# OSGEMM with the osCore A input held in an nb_slots-slot async ring (refilled from L3
# during compute, paced by R10) and tiled along dInner. Layout (see src/main.c):
#   [ A ring (nb_slots slots) | B tile | D full ]
# Only nb_slots A L-tiles are resident; B is one dInner tile; D is full-resident (the
# osCore writes tile slices into it). Tiling keeps the footprint flat in seqLen.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (  # type: ignore[import]
    FP8, MemoryReport, sequential_bytes, run_model,
)


def build_report(params: dict) -> MemoryReport:
    seqLen = params["dim0"]
    dModel = params["dim1"]
    dInner = params["dim2"]
    nb = params["nb_tiles"]
    nb_l_tiles = params["nb_l_tiles"]
    nb_slots = params["nb_slots"]

    report = MemoryReport("osgemm-tiled-async", {
        "dim0": seqLen, "dim1": dModel, "dim2": dInner,
        "nb_tiles": nb, "nb_l_tiles": nb_l_tiles, "nb_slots": nb_slots,
    })

    len_a_l_tile = (seqLen // nb_l_tiles) * dModel * FP8 // 8
    len_b_tile = (dModel * dInner * FP8 // 8) // nb
    len_d = seqLen * dInner * FP8 // 8

    live_bufs = [
        (f"A ring ({nb_slots}x{len_a_l_tile}B)", nb_slots * len_a_l_tile),
        ("B tile", len_b_tile),
        ("D (full, resident)", len_d),
    ]
    report.add_section("Resident during compute", live_bufs)
    report.add_peak("OSGEMM async (all buffers live)", sequential_bytes([s for _, s in live_bufs]))
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
