#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `fft` app (non-tiled).
# Three-phase FFT: partition1 (ISGEMM), hadamard+reorder (SIMD), partition2 (ISGEMM).
# All buffers sequential in TCDM — no tiling, no L3 staging.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8, BF16, D_INNER_UNROLL, pad_to_unroll,
    MemoryReport, sequential_bytes, run_model,
)


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    L1 = params["L1"]
    L2 = params["L2"]
    assert L1 * L2 == L, f"L1*L2={L1*L2} != seqLen={L}"
    L1_padded = pad_to_unroll(L1, D_INNER_UNROLL)
    L2_padded = pad_to_unroll(L2, D_INNER_UNROLL)

    report = MemoryReport("fft", {
        "seqLen": L, "dModel": dModel, "L1": L1, "L2": L2,
        "L1_padded": L1_padded, "L2_padded": L2_padded,
    })

    bufs = [
        ("weight1",             2 * L1 * L1_padded * FP8 // 8),
        ("weight2",             2 * L2 * 2 * L2_padded * FP8 // 8),
        ("in",                  L * dModel * FP8 // 8),
        ("partition1_out",      2 * L * dModel * BF16 // 8),
        ("twiddles",            2 * L * FP8 // 8),
        ("hadamard_out",        2 * L * dModel * FP8 // 8),
        ("hadamard_reordered",  2 * L * dModel * FP8 // 8),
        ("partition2_out",      2 * L * dModel * BF16 // 8),
    ]
    report.add_section("All buffers (sequential in TCDM)", bufs)
    total = sequential_bytes([s for _, s in bufs])
    report.add_peak("FFT (all phases)", total)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
