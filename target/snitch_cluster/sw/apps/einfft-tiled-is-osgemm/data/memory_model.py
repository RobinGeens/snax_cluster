#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `einfft-tiled-is-osgemm` app.
#
# Dual-core (OS + IS in parallel) 2-layer complex EinFFT MLP, 4 branches.
# No N-tiling (the IS_OSGEMM shared CSRs make OS-N-tile == IS-K-tile, so both
# matmuls run full (L, dPerB) @ (dPerB, dPerB)). Per (layer, branch):
#   REAL side -> OS-core (rr, ii) FP8 ConvFormat, fused like einfft.
#   IMAG side -> IS-core (ri, ir) raw BF16 flattenCD, accumulated into one psum P.
#   Shared SIMD fuse over the full L*dPerB per side.
# Only ONE branch is resident at a time -> peak = per-branch footprint.
# Mirrors the per_branch_resident sum in datagen.py.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8, BF16, SEQ_LEN_UNROLL, D_INNER_UNROLL, CONV_UNROLL, align64,
    MemoryReport, run_model,
)

NB_BRANCHES = 4


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]

    Mu = SEQ_LEN_UNROLL   # 16
    Nu = D_INNER_UNROLL   # 24
    cu = CONV_UNROLL      # 4

    assert dModel % NB_BRANCHES == 0
    dPerB = dModel // NB_BRANCHES
    assert dPerB % Nu == 0, f"dPerB ({dPerB}) must be a multiple of dInnerUnroll ({Nu})"

    report = MemoryReport("einfft-tiled-is-osgemm", {
        "seqLen": L, "dModel": dModel, "dPerB": dPerB,
    })

    len_x_branch   = L * dPerB * FP8 // 8
    len_w_branch   = dPerB * dPerB * FP8 // 8
    len_d          = L * dPerB * FP8 // 8       # OS scratch (rr/ii) ConvFormat FP8
    len_cd         = L * dPerB * BF16 // 8      # IS psum P (ri/ir) BF16
    len_bf16       = L * dPerB * BF16 // 8      # real SIMD staging
    len_out_branch = L * dPerB * FP8 // 8
    l2_count       = L // Mu
    groups_per_d3  = Nu // cu
    len_bias_mini_branch = (dPerB // Nu) * l2_count * groups_per_d3 * 16 * BF16 // 8

    inputs = [
        ("x_re (flatA)",        len_x_branch),
        ("x_im (flatA)",        len_x_branch),
        ("x_re_conv (Conv)",    len_x_branch),
        ("x_im_conv (Conv)",    len_x_branch),
    ]
    report.add_section("Per-branch inputs (x, FP8)", inputs)

    weights = [
        ("W_re (OS)",     len_w_branch),
        ("W_im (OS)",     len_w_branch),
        ("W_re_is (IS)",  len_w_branch),
        ("W_im_is (IS)",  len_w_branch),
    ]
    report.add_section("Per-branch weights (FP8)", weights)

    scratch = [
        ("rr (OS scratch, FP8)",  len_d),
        ("ii (OS scratch, FP8)",  len_d),
        ("P (IS psum, BF16)",     len_cd),
        ("bf16_a (SIMD staging)", len_bf16),
        ("bf16_b (SIMD staging)", len_bf16),
    ]
    report.add_section("Scratch / staging", scratch)

    biases_out = [
        ("b_re mini (real, BF16)", len_bias_mini_branch),
        ("out_re (FP8)",           len_out_branch),
        ("out_im (FP8)",           len_out_branch),
    ]
    report.add_section("Bias + outputs", biases_out)

    per_branch_resident = (
        align64(len_x_branch) * 4
        + align64(len_w_branch) * 4
        + align64(len_d) * 2
        + align64(len_cd) * 1
        + align64(len_bf16) * 2
        + align64(len_bias_mini_branch) * 1
        + align64(len_out_branch) * 2
    )

    # L3 staging (not counted in TCDM).
    len_x   = NB_BRANCHES * len_x_branch
    len_w   = NB_BRANCHES * len_w_branch
    len_out = NB_BRANCHES * len_out_branch
    l3_bufs = [
        ("x_real/imag (+conv) × 2 layers (L3)",    8 * len_x),
        ("weight_real/imag (OS+IS) × 2 layers (L3)", 8 * len_w),
        ("output_real/imag × 2 layers (L3)",       4 * len_out),
    ]
    report.add_section("L3 staging (not counted in TCDM)", l3_bufs)

    report.add_peak("Per-branch resident", per_branch_resident)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
