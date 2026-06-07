#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `main-tiled` app.
#
# Two-phase Mamba with dInner-axis tiling and L3 ping-pong.
#
# Shared FULL (always live across P1 and P2):
#   oscore_in, iscore_out_P1 (= dt_in for P2), iscore_out_P2
#
# Phase 1 ping-pong (×2 each):
#   oscore_weight_tile, conv_weight_tile, conv_bias_tile,
#   iscore_weight_tile, conv_out_tile (DMA to L3)
#
# Phase 2 ping-pong (×2 each, overlays P1 scratch):
#   oscore_weight_tile, dt_weight_1_tile, dt_weight_2_tile, dt_bias_tile,
#   A_tile, D_tile, iscore_weight_tile, x_tile (DMA from L3),
#   z_tile (DMA to L3), y_tile (DMA to L3)
#
# Peak = shared + max(P1_pp, P2_pp)

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8, BF16, align64,
    derive_mamba_params, MemoryReport, pingpong_bytes, sequential_bytes, run_model,
    bc_pad_iscore_out_bytes,
)


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    nb = params["nb_tiles"]
    derived = derive_mamba_params(params)
    dInner = derived["dInner"]
    xProjDim = derived["xProjDim"]
    dtRank = derived["dtRank"]
    dConv = derived["dConv"]
    dState = derived["dState"]
    dtRankUnroll = 6

    report = MemoryReport("main-tiled", {
        "seqLen": L, "dModel": dModel, "dInner": dInner,
        "xProjDim": xProjDim, "nb_tiles": nb,
    })

    # Shared FULL buffers.
    # The P1 isCore psum (BF16) is NO LONGER full-resident: every P1 tile is NO_REQUANT, so the psum rides
    # an nb_slots-slot async ring to L3 (full copy in L3). What stays resident is dt_in -- P2's transposed
    # FP8 dt+BC (half the BF16 psum), loaded off-chip after P1 -- and the P1 ring OVERLAYS dt_in's region
    # (disjoint phases), so this buffer is just max(dt_in, ring).
    nb_l = params["nb_l_tiles"]
    nb_slots = params["nb_slots"]
    len_psum_full = bc_pad_iscore_out_bytes(params, L, xProjDim)  # BF16 psum (full), lives in L3
    len_dt_in = len_psum_full // 2                                # FP8 transposed dt+BC (P2 input)
    len_p1_ring = nb_slots * (len_psum_full // nb_l)              # resident ring (P1)
    len_dt_in_region = max(len_dt_in, len_p1_ring)               # ring overlays dt_in
    len_oscore_in = L * dModel * FP8 // 8
    len_iscore_out_P2 = L * dModel * BF16 // 8

    shared_bufs = [
        ("oscore_in",                          len_oscore_in),
        ("dt_in / P1 ring (overlay)",          len_dt_in_region),
        ("iscore_out_P2",                      len_iscore_out_P2),
    ]
    report.add_section("Shared FULL (always live)", shared_bufs)
    shared_bytes = sequential_bytes([s for _, s in shared_bufs])

    # Phase 1 ping-pong tiles
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
    l3_bufs = [
        ("conv_out (L3)",          len_conv_out),
        ("z (L3)",                 len_z),
        ("y (L3)",                 len_y),
        ("iscore_out_P1 psum (L3)", len_psum_full),  # full BF16 psum; ring spills/reloads here
    ]
    report.add_section("L3 staging (not counted in TCDM)", l3_bufs)

    p1_peak = align64(shared_bytes) + p1_pp
    p2_peak = align64(shared_bytes) + p2_pp
    report.add_peak("Phase 1 (shared + P1 PP)", p1_peak)
    report.add_peak("Phase 2 (shared + P2 PP)", p2_peak)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
