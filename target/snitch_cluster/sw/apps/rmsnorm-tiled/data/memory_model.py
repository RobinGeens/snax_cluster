#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `rmsnorm-tiled` app.
# Double-buffered L-tile pipeline: 2 x-slots (DMA touches one while the SIMD core works the
# other), a single per-tile rms scratch (compute-local), plus resident weight + constants.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    BF16, SIMD_LANES_BF16,
    MemoryReport, sequential_bytes, run_model,
)


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    D = params["dModel"]
    nb_tiles = params["nb_tiles"]
    Lt = L // nb_tiles

    report = MemoryReport("rmsnorm-tiled", {"seqLen": L, "dModel": D, "nb_tiles": nb_tiles, "L_tile": Lt})

    bufs = [
        ("weight",    SIMD_LANES_BF16 * D * BF16 // 8),
        ("d_inverse", SIMD_LANES_BF16 * BF16 // 8),
        ("ones",      SIMD_LANES_BF16 * BF16 // 8),
        ("rms",       Lt * BF16 // 8),
        ("x_slot0",   Lt * D * BF16 // 8),
        ("x_slot1",   Lt * D * BF16 // 8),
    ]
    report.add_section("Double-buffered L-tile pipeline", bufs)
    total = sequential_bytes([s for _, s in bufs])
    report.add_peak("RMSNorm tiled", total)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
