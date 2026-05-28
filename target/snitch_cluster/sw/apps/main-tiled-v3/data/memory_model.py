#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `main-tiled-v3` app.
#
# Loop order: dInner OUTER, N_tile INNER (vs. v2 which is N OUTER, dInner INNER).
# Psum [L, N_tile] is staged through L3: at every (dInner, n) the partial psum
# is DMA-loaded in and (after compute) spilled back out. Inner N loop has a
# 2-slot ping-pong on the psum buffer so DMA-in[n], compute[n-1] and
# DMA-out[n-2] can run in parallel.
#
# Shared FULL: oscore_in, iscore_out P1/P2 ping-pong (2 slots, sized for the
#              larger of P1 / P2 psum tile).
# Phase 1 ping-pong (×2 each): oscore_weight_tile, conv_weight_tile,
#   conv_bias_tile, iscore_weight_tile (FULL N per K-row), conv_out_tile.
# Phase 2 same shape (overlays P1 PP region).
#
# Peak = shared + max(P1_pp, P2_pp)

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8, BF16, align64,
    derive_mamba_params, MemoryReport, pingpong_bytes, sequential_bytes, run_model,
)


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    nb = params["nb_tiles"]
    nn = params["nb_n_tiles"]
    derived = derive_mamba_params(params)
    dInner = derived["dInner"]
    xProjDim = derived["xProjDim"]
    dtRank = derived["dtRank"]
    dConv = derived["dConv"]
    dState = derived["dState"]
    dtRankUnroll = 6

    xProjDim_tile = xProjDim // nn
    dModel_tile = dModel // nn

    report = MemoryReport("main-tiled-v3", {
        "seqLen": L, "dModel": dModel, "dInner": dInner,
        "xProjDim": xProjDim, "nb_tiles": nb, "nb_n_tiles": nn,
    })

    # Shared FULL buffers (per N_tile iteration)
    # BankTransposer gated on isCoreOutIsFinal: same buffer for psum and final.
    # P1 and P2 overlay (different phase).
    len_oscore_in = L * dModel * FP8 // 8
    len_iscore_psum_P1 = L * xProjDim_tile * BF16 // 8
    len_iscore_psum_P2 = L * dModel_tile * BF16 // 8
    iscore_shared = max(len_iscore_psum_P1, len_iscore_psum_P2)

    shared_bufs = [
        ("oscore_in",                                 len_oscore_in),
        ("iscore_out tile slot 0 (P1/P2 ping-pong)",  iscore_shared),
        ("iscore_out tile slot 1 (P1/P2 ping-pong)",  iscore_shared),
    ]
    report.add_section("Shared FULL (always live)", shared_bufs)
    shared_bytes = sequential_bytes([s for _, s in shared_bufs])

    # Phase 1 ping-pong tiles (iscore_weight still has full xProjDim per dInner tile)
    len_oscore_weight = dModel * dInner * FP8 // 8
    len_conv_weight = dInner * dConv * FP8 // 8
    len_conv_bias = dInner * FP8 // 8
    len_conv_out = L * dInner * FP8 // 8
    len_iscore_weight_P1 = dInner * xProjDim * FP8 // 8

    p1_tiles = [
        ("oscore_weight_tile",  len_oscore_weight // nb),
        ("conv_weight_tile",    len_conv_weight // nb),
        ("conv_bias_tile",      len_conv_bias // nb),
        ("iscore_weight_tile",  len_iscore_weight_P1 // nb),
        ("conv_out_tile",       len_conv_out // nb),
    ]
    report.add_section("Phase 1 ping-pong (×2 each)", p1_tiles)
    p1_pp = pingpong_bytes([s for _, s in p1_tiles])

    # Phase 2 ping-pong tiles (overlays P1 scratch)
    len_z = L * dInner * FP8 // 8
    len_dt_w1 = dInner * (dtRank // dtRankUnroll) * dConv * FP8 // 8
    len_dt_w2 = dInner * (dtRank // dtRankUnroll) * (dtRankUnroll - dConv) * FP8 // 8
    len_dt_bias = dInner * FP8 // 8
    len_A = dInner * dState * FP8 // 8
    len_D = dInner * FP8 // 8
    len_x = len_conv_out
    len_y = L * dInner * FP8 // 8
    len_iscore_weight_P2 = dModel * dInner * FP8 // 8

    p2_tiles = [
        ("oscore_weight_tile",  len_oscore_weight // nb),
        ("dt_weight_1_tile",    len_dt_w1 // nb),
        ("dt_weight_2_tile",    len_dt_w2 // nb),
        ("dt_bias_tile",        len_dt_bias // nb),
        ("A_tile",              len_A // nb),
        ("D_tile",              len_D // nb),
        ("iscore_weight_tile",  len_iscore_weight_P2 // nb),
        ("x_tile",              len_conv_out // nb),
        ("z_tile",              len_z // nb),
        ("y_tile",              len_y // nb),
    ]
    report.add_section("Phase 2 ping-pong (×2 each, overlays P1)", p2_tiles)
    p2_pp = pingpong_bytes([s for _, s in p2_tiles])

    # L3 staging
    len_iscore_out_P1_full = L * xProjDim * BF16 // 8
    len_iscore_out_P2_full = L * dModel * BF16 // 8
    l3_bufs = [
        ("conv_out (L3)",                   len_conv_out),
        ("z (L3)",                          len_z),
        ("y (L3)",                          len_y),
        ("iscore_out_P1 psum staging (L3)", len_iscore_out_P1_full),
        ("iscore_out_P1 final (L3)",        len_iscore_out_P1_full),
        ("iscore_out_P2 psum staging (L3)", len_iscore_out_P2_full),
        ("iscore_out_P2 final (L3)",        len_iscore_out_P2_full),
    ]
    report.add_section("L3 staging (not counted in TCDM)", l3_bufs)

    p1_peak = align64(shared_bytes) + p1_pp
    p2_peak = align64(shared_bytes) + p2_pp
    report.add_peak("Phase 1 (shared + P1 PP)", p1_peak)
    report.add_peak("Phase 2 (shared + P2 PP)", p2_peak)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
