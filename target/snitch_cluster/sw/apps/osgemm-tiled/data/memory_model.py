#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `osgemm-tiled` app.
#
# Single OSGEMM tiled along dInner (dim2). No phases: every buffer is resident at
# once for the whole pipelined loop. Input A is loaded once and shared across tiles
# (full resident). B and D are double-buffered ping-pong tiles. The transfer_out
# destination D_full stays in TCDM (it emulates the off-chip output, which in a real
# deployment would live in L3). Layout: see src/main.c.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (  # type: ignore[import]
    FP8, MemoryReport, pingpong_bytes, sequential_bytes, run_model,
)


def build_report(params: dict) -> MemoryReport:
    seqLen = params["dim0"]
    dModel = params["dim1"]
    dInner = params["dim2"]
    nb = params["nb_tiles"]

    report = MemoryReport("osgemm-tiled", {
        "dim0": seqLen, "dim1": dModel, "dim2": dInner, "nb_tiles": nb,
    })

    # FULL resident: A (shared across tiles) and D_full (emulated off-chip output)
    len_a = seqLen * dModel * FP8 // 8
    len_d = seqLen * dInner * FP8 // 8
    full_bufs = [
        ("A (osgemm A, shared)", len_a),
        ("D_full (emulated off-chip out)", len_d),
    ]
    report.add_section("FULL resident", full_bufs)
    full_bytes = sequential_bytes([s for _, s in full_bufs])

    # Ping-pong tiles (×2 each), tiled along dInner
    len_b_tile = (dModel * dInner * FP8 // 8) // nb
    len_d_tile = len_d // nb
    pp_tiles = [
        ("B_tile", len_b_tile),
        ("D_tile", len_d_tile),
    ]
    report.add_section("Ping-pong (×2 each, tiled along dInner)", pp_tiles)
    pp_bytes = pingpong_bytes([s for _, s in pp_tiles])

    report.add_peak("Resident (full + ping-pong)", full_bytes + pp_bytes)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
