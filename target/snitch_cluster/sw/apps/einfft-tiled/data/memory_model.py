#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `einfft-tiled` app.
#
# 2-layer complex EinFFT MLP with 4 branches. Per (layer, branch):
#   DMA x_re, x_im, bias_re_bcast, bias_im_bcast into TCDM.
#   Weight tiles ping-ponged (W_re, W_im × 2).
#   4 OSGEMM scratch buffers (rr, ii, ri, ir) at FULL per-branch size.
#   2 BF16 staging buffers for SIMD fuse.
#   Output re/im DMA'd out to L3.
#
# Only ONE branch is resident at a time — peak = per-branch footprint.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8, BF16, D_INNER_UNROLL, SEQ_LEN_UNROLL, CONV_UNROLL, align64,
    MemoryReport, run_model,
)

NB_BRANCHES = 4


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    nb_tiles = params["nb_tiles"]

    Nu = D_INNER_UNROLL   # 24
    Mu = SEQ_LEN_UNROLL   # 16
    cu = CONV_UNROLL      # 4

    assert dModel % NB_BRANCHES == 0
    dPerB = dModel // NB_BRANCHES
    dPerB_t = dPerB // nb_tiles

    report = MemoryReport("einfft-tiled", {
        "seqLen": L, "dModel": dModel, "nb_tiles": nb_tiles,
        "dPerB": dPerB, "dPerB_t": dPerB_t,
    })

    len_x_branch = L * dPerB * FP8 // 8
    len_w_branch = dPerB * dPerB * FP8 // 8
    len_w_branch_tile = len_w_branch // nb_tiles
    # Per-tile sizes — the fuse now runs one N-tile at a time, so scratch /
    # BF16 staging / output only need to hold a single tile (L × dPerB_t).
    len_d_tile = L * dPerB_t * FP8 // 8
    len_bf16 = L * dPerB_t * BF16 // 8
    # Mini-expanded bias (conv-walk order), FULL per branch, double-buffered.
    len_bias_mini_branch = (dPerB // Nu) * (L // Mu) * (Nu // cu) * 16 * BF16 // 8

    bufs = [
        # Inputs (FULL per branch: x is reused by every N-tile)
        ("x_re_b",          len_x_branch),
        ("x_im_b",          len_x_branch),
        ("bias_re_mini[0]", len_bias_mini_branch),
        ("bias_re_mini[1]", len_bias_mini_branch),
        ("bias_im_mini[0]", len_bias_mini_branch),
        ("bias_im_mini[1]", len_bias_mini_branch),
        # Outputs (TILE, ping-pong for DMA overlap)
        ("out_re_pp[0]",    len_d_tile),
        ("out_re_pp[1]",    len_d_tile),
        ("out_im_pp[0]",    len_d_tile),
        ("out_im_pp[1]",    len_d_tile),
    ]
    report.add_section("Per-branch inputs/outputs", bufs)

    pp_bufs = [
        ("W_re_pp[0]",     len_w_branch_tile),
        ("W_re_pp[1]",     len_w_branch_tile),
        ("W_im_pp[0]",     len_w_branch_tile),
        ("W_im_pp[1]",     len_w_branch_tile),
    ]
    report.add_section("Weight ping-pong (×2 re, ×2 im)", pp_bufs)

    scratch_bufs = [
        ("rr (OSGEMM scratch)",  len_d_tile),
        ("ii (OSGEMM scratch)",  len_d_tile),
        ("ri (OSGEMM scratch)",  len_d_tile),
        ("ir (OSGEMM scratch)",  len_d_tile),
    ]
    report.add_section("OSGEMM scratch (TILE)", scratch_bufs)

    bf16_bufs = [
        ("bf16_a",  len_bf16),
        ("bf16_b",  len_bf16),
    ]
    report.add_section("BF16 staging (TILE)", bf16_bufs)

    per_branch_resident = (
        align64(len_x_branch) * 2
        + align64(len_bias_mini_branch) * 2 * 2  # bias_re/im mini, double-buffered
        + align64(len_d_tile) * 2 * 2            # out_re_pp[2] + out_im_pp[2]
        + align64(len_w_branch_tile) * 2 * 2     # W_re_pp[2] + W_im_pp[2]
        + align64(len_d_tile) * 4                # rr, ii, ri, ir
        + align64(len_bf16) * 2                  # bf16_a, bf16_b
    )

    # L3 staging
    len_x = NB_BRANCHES * len_x_branch
    len_out = NB_BRANCHES * len_x_branch          # output is L*dPerB FP8 per branch, like x
    l3_bufs = [
        ("x_real/imag × 2 layers (L3)",         4 * len_x),
        ("bias_mini × 4 (L3)",                  4 * NB_BRANCHES * len_bias_mini_branch),
        ("output_real/imag × 2 layers (L3)",     4 * len_out),
        ("weight_real/imag × 2 layers (L3)",     4 * NB_BRANCHES * len_w_branch),
    ]
    report.add_section("L3 staging (not counted in TCDM)", l3_bufs)

    report.add_peak("Per-branch resident", per_branch_resident)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
