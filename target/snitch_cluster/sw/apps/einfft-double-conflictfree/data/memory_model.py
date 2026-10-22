#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8,
    BF16,
    SEQ_LEN_UNROLL,
    D_INNER_UNROLL,
    CONV_UNROLL,
    align64,
    MemoryReport,
    run_model,
)

NB_BRANCHES = 4
SKIP = 2  # skip-128 layout doubles each GEMM buffer's address footprint


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]

    Mu = SEQ_LEN_UNROLL  # 16
    Nu = D_INNER_UNROLL  # 24
    cu = CONV_UNROLL  # 4

    assert dModel % NB_BRANCHES == 0
    dPerB = dModel // NB_BRANCHES
    assert dPerB % Nu == 0, f"dPerB ({dPerB}) must be a multiple of dInnerUnroll ({Nu})"

    report = MemoryReport(
        "einfft-double-conflictfree",
        {
            "seqLen": L,
            "dModel": dModel,
            "dPerB": dPerB,
        },
    )

    len_x_branch = L * dPerB * FP8 // 8
    len_w_branch = dPerB * dPerB * FP8 // 8
    len_d = L * dPerB * FP8 // 8  # OS scratch (rr/ii) ConvFormat FP8
    len_cd = L * dPerB * BF16 // 8  # IS psum P (ri/ir) BF16
    len_bf16 = L * dPerB * BF16 // 8  # real SIMD staging
    len_out_branch = L * dPerB * FP8 // 8
    l2_count = L // Mu
    groups_per_d3 = Nu // cu
    len_bias_mini_branch = (dPerB // Nu) * l2_count * groups_per_d3 * 16 * BF16 // 8

    # OS half (banks 0-15): x_re/x_im (flatA), W_re/W_im, rr/ii scratch -- all skip-128.
    # DMA'd inputs are double-buffered (x2); rr/ii scratch is single-buffered.
    os_half = [
        ("x_re (flatA, skip-128, x2)", 2 * SKIP * len_x_branch),
        ("x_im (flatA, skip-128, x2)", 2 * SKIP * len_x_branch),
        ("W_re (OS, skip-128, x2)", 2 * SKIP * len_w_branch),
        ("W_im (OS, skip-128, x2)", 2 * SKIP * len_w_branch),
        ("rr (OS scratch, skip-128)", SKIP * len_d),
        ("ii (OS scratch, skip-128)", SKIP * len_d),
    ]
    report.add_section("OS half, banks 0-15 (skip-128)", os_half)

    # IS half (banks 16-31): x_re_conv/x_im_conv (Conv), W_re_is/W_im_is, psum P -- all skip-128.
    is_half = [
        ("x_re_conv (Conv, skip-128, x2)", 2 * SKIP * len_x_branch),
        ("x_im_conv (Conv, skip-128, x2)", 2 * SKIP * len_x_branch),
        ("W_re_is (IS, skip-128, x2)", 2 * SKIP * len_w_branch),
        ("W_im_is (IS, skip-128, x2)", 2 * SKIP * len_w_branch),
        ("P (IS psum, BF16, skip-128, x2)", 2 * SKIP * len_cd),
    ]
    report.add_section("IS half, banks 16-31 (skip-128)", is_half)

    # SIMD-only buffers stay contiguous; bf16 staging single-buffered, bias/outputs double-buffered.
    simd = [
        ("bf16_a (SIMD staging)", len_bf16),
        ("b_re mini (real bias, BF16, x2)", 2 * len_bias_mini_branch),
        ("out_re (FP8, x2)", 2 * len_out_branch),
        ("out_im (FP8, x2)", 2 * len_out_branch),
    ]
    report.add_section("SIMD staging / bias / outputs (contiguous)", simd)

    # The OS and IS heaps overlap (same base, IS at +128), so the partitioned resident set is
    # 2*max(OS, IS), not their sum. The SIMD region is contiguous and stacks on top.
    os_span = (
        align64(SKIP * len_x_branch) * 4  # x_re/x_im x2
        + align64(SKIP * len_w_branch) * 4  # W_re/W_im x2
        + align64(SKIP * len_d) * 2  # rr/ii (single)
    )
    is_span = (
        align64(SKIP * len_x_branch) * 4  # x_*_conv x2
        + align64(SKIP * len_w_branch) * 4  # W_*_is x2
        + align64(SKIP * len_cd) * 2  # P x2
    )
    simd_span = align64(len_bf16) + align64(len_bias_mini_branch) * 2 + align64(len_out_branch) * 2 * 2
    per_branch_resident = max(os_span, is_span) + simd_span

    # L3 staging (not counted in TCDM).
    len_x = NB_BRANCHES * len_x_branch
    len_w = NB_BRANCHES * len_w_branch
    len_out = NB_BRANCHES * len_out_branch
    l3_bufs = [
        ("x_real/imag (+conv) × 2 layers (L3)", 8 * len_x),
        ("weight_real/imag (OS+IS) × 2 layers (L3)", 8 * len_w),
        ("output_real/imag × 2 layers (L3)", 4 * len_out),
    ]
    report.add_section("L3 staging (not counted in TCDM)", l3_bufs)

    report.add_peak("Per-branch resident", per_branch_resident)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
