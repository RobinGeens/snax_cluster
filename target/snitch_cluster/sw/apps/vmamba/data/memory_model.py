#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `vmamba` app (non-tiled).
#
# Per-direction: Phase 1 + Phase 2 (same layout as `main`).
# After all 4 directions: cross-merge (2 buffers overlay P1+P2 base)
# then RMSNorm (buffers placed after P1+P2 end).
#
# Peak = max(fused P1+P2, merge, rmsnorm_end)

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8, BF16, SIMD_LANES_BF16, align64,
    derive_mamba_params, MemoryReport, sequential_bytes, run_model,
)


def build_report(params: dict) -> MemoryReport:
    H = params["H"]
    W = params["W"]
    L = H * W
    dModel = params["dModel"]
    derived = derive_mamba_params(params)
    dInner = derived["dInner"]
    xProjDim = derived["xProjDim"]
    dtRank = derived["dtRank"]
    dConv = derived["dConv"]
    dState = derived["dState"]
    dtRankUnroll = 6

    report = MemoryReport("vmamba", {
        "H": H, "W": W, "seqLen": L, "dModel": dModel,
        "dInner": dInner, "xProjDim": xProjDim,
    })

    # Phase 1 buffers (sequential)
    p1_bufs = [
        ("oscore_in",      L * dModel * FP8 // 8),
        ("oscore_weight",  dModel * dInner * FP8 // 8),
        ("conv_weight",    dInner * dConv * FP8 // 8),
        ("conv_bias",      dInner * FP8 // 8),
        ("conv_out",       L * dInner * FP8 // 8),
        ("iscore_weight",  dInner * xProjDim * FP8 // 8),
        ("iscore_out",     L * xProjDim * BF16 // 8),
    ]
    report.add_section("Phase 1 (per-direction, sequential in TCDM)", p1_bufs)
    p1_total = sequential_bytes([s for _, s in p1_bufs])

    # Phase 2 buffers (sequential, base after P1 iscore_out)
    p2_bufs = [
        ("oscore_in",      L * dModel * FP8 // 8),
        ("oscore_weight",  dModel * dInner * FP8 // 8),
        ("z",              L * dInner * FP8 // 8),
        ("dt_BC",          L * xProjDim * FP8 // 8),
        ("dt_weight_1",    dInner * (dtRank // dtRankUnroll) * dConv * FP8 // 8),
        ("dt_weight_2",    dInner * (dtRank // dtRankUnroll) * (dtRankUnroll - dConv) * FP8 // 8),
        ("dt_bias",        dInner * FP8 // 8),
        ("x",              L * dInner * FP8 // 8),
        ("A",              dInner * dState * FP8 // 8),
        ("D",              dInner * FP8 // 8),
        ("y",              L * dInner * FP8 // 8),
        ("iscore_weight",  dModel * dInner * FP8 // 8),
        ("iscore_out",     L * dModel * BF16 // 8),
    ]
    report.add_section("Phase 2 (per-direction, starts after P1 iscore_out)", p2_bufs)
    p2_total = sequential_bytes([s for _, s in p2_bufs])
    fused_p1p2 = p1_total + p2_total

    # Cross-merge: 2 buffers overlaying TCDM base
    merge_bufs = [
        ("merge_a",  L * dInner * FP8 // 8),
        ("merge_b",  L * dInner * FP8 // 8),
    ]
    report.add_section("Cross-merge (overlays P1+P2 base)", merge_bufs)
    merge_total = sequential_bytes([s for _, s in merge_bufs])

    # RMSNorm: placed after P1+P2 end (after the full P1+P2 region, not after merge)
    D = dInner
    rms_bufs = [
        ("rms_x_bf16",  L * D * BF16 // 8),
        ("rms_const",   SIMD_LANES_BF16 * BF16 // 8),
        ("rms_weight",  SIMD_LANES_BF16 * D * BF16 // 8),
        ("rms_vec",     L * BF16 // 8),
    ]
    report.add_section("RMSNorm (placed after P1+P2 region)", rms_bufs)
    rms_base = align64(fused_p1p2)
    rms_total = sequential_bytes([s for _, s in rms_bufs])
    rms_peak = rms_base + rms_total

    report.add_peak("Fused P1+P2 (per-direction)", fused_p1p2)
    report.add_peak("Cross-merge", merge_total)
    report.add_peak("RMSNorm", rms_peak)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
