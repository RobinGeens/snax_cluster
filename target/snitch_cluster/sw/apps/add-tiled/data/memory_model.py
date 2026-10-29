#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `add-tiled` app.
# Double-buffered tile pipeline: 2 a-slots (the sum is written in place) + 2 b-slots.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8,
    MemoryReport, sequential_bytes, run_model,
)


def build_report(params: dict) -> MemoryReport:
    N = params["numElem"]
    nb_tiles = params["nb_tiles"]
    tile = N * FP8 // 8

    report = MemoryReport("add-tiled", {"numElem": N, "nb_tiles": nb_tiles, "tile_bytes": tile})

    bufs = [
        ("a_slot0", tile),
        ("a_slot1", tile),
        ("b_slot0 (+32 B bank offset)", tile + 32),
        ("b_slot1", tile),
    ]
    report.add_section("Double-buffered tile pipeline", bufs)
    report.add_peak("Add tiled", sequential_bytes([s for _, s in bufs]))
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
