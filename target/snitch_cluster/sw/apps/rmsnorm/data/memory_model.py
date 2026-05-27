#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `rmsnorm` app.
# Single-phase SIMD chain: all buffers sequential in TCDM.

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

    report = MemoryReport("rmsnorm", {"seqLen": L, "dModel": D})

    bufs = [
        ("x",         L * D * BF16 // 8),
        ("d_inverse", SIMD_LANES_BF16 * BF16 // 8),
        ("ones",      SIMD_LANES_BF16 * BF16 // 8),
        ("weight",    SIMD_LANES_BF16 * D * BF16 // 8),
        ("rms",       L * BF16 // 8),
    ]
    report.add_section("Single phase (all sequential in TCDM)", bufs)
    total = sequential_bytes([s for _, s in bufs])
    report.add_peak("RMSNorm", total)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
