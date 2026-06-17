#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `batchnorm` app.
#
# Folded BatchNorm + ReLU: a single SIMD per-channel-affine pass, no tiling. The
# activations x, the lane-duplicated scale and shift vectors, and the output are all
# resident at once, so peak L1 = the sequentially-packed sum of all four. Layout:
# see src/batchnorm.c and data/datagen.py.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (  # type: ignore[import]
    BF16, SIMD_LANES_BF16, MemoryReport, sequential_bytes, run_model,
)


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    D = params["channels"]
    lanes = params.get("simdLanes_bf16", SIMD_LANES_BF16)

    report = MemoryReport("batchnorm", {"seqLen": L, "channels": D})

    bufs = [
        ("x",     L * D * BF16 // 8),
        ("scale", lanes * D * BF16 // 8),  # per-channel scalar, duplicated over lanes
        ("shift", lanes * D * BF16 // 8),
        ("out",   L * D * BF16 // 8),
    ]
    report.add_section("FULL resident (single SIMD pass)", bufs)
    report.add_peak("Resident (x + scale + shift + out)", sequential_bytes([s for _, s in bufs]))
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
