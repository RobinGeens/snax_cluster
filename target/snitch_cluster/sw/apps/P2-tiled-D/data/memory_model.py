#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for P2-tiled-D
#
# FULL (always live):
#   oscore_in, dt_BC (= iscore_out_P1, BC bank-padded), iscore_out_P2
#
# Phase 2 ping-pong (×2 each):
#   oscore_weight_tile, dt_weight_1_tile, dt_weight_2_tile, dt_bias_tile,
#   A_tile, D_tile, iscore_weight_tile, x_tile (from L3),
#   z_tile (to L3), y_tile (to L3)
#
# Peak = align64(FULL) + P2 ping-pong

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8,
    BF16,
    align64,
    derive_mamba_params,
    MemoryReport,
    pingpong_bytes,
    sequential_bytes,
    run_model,
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

    report = MemoryReport(
        "P2-tiled-D",
        {
            "seqLen": L,
            "dModel": dModel,
            "dInner": dInner,
            "xProjDim": xProjDim,
            "nb_tiles": nb,
        },
    )

    # FULL buffers (preloaded from golden, live for the whole run)
    len_oscore_in = L * dModel * FP8 // 8
    len_dt_BC = bc_pad_iscore_out_bytes(params, L, xProjDim)  # = iscore_out_P1, BC bank pad §5.5
    len_iscore_out_P2 = L * dModel * BF16 // 8

    full_bufs = [
        ("oscore_in", len_oscore_in),
        ("dt_BC (= dt_in)", len_dt_BC),
        ("iscore_out_P2", len_iscore_out_P2),
    ]
    report.add_section("FULL (always live)", full_bufs)
    full_bytes = sequential_bytes([s for _, s in full_bufs])

    # Phase 2 ping-pong tiles
    len_oscore_weight = dModel * dInner * FP8 // 8
    len_dt_w1 = dInner * (dtRank // dtRankUnroll) * dConv * FP8 // 8
    len_dt_w2 = dInner * (dtRank // dtRankUnroll) * (dtRankUnroll - dConv) * FP8 // 8
    len_dt_bias = dInner * FP8 // 8
    len_A = dInner * dState * FP8 // 8
    len_D = dInner * FP8 // 8
    len_x = L * dInner * FP8 // 8
    len_z = L * dInner * FP8 // 8
    len_y = L * dInner * FP8 // 8
    len_iscore_weight = dModel * dInner * FP8 // 8

    p2_tiles = [
        ("oscore_weight_tile", len_oscore_weight // nb),
        ("dt_weight_1_tile", len_dt_w1 // nb),
        ("dt_weight_2_tile", len_dt_w2 // nb),
        ("dt_bias_tile", len_dt_bias // nb),
        ("A_tile", len_A // nb),
        ("D_tile", len_D // nb),
        ("iscore_weight_tile", len_iscore_weight // nb),
        ("x_tile", len_x // nb),
        ("z_tile", len_z // nb),
        ("y_tile", len_y // nb),
    ]
    report.add_section("Phase 2 ping-pong (×2 each)", p2_tiles)
    p2_pp = pingpong_bytes([s for _, s in p2_tiles])

    # L3 staging (not counted in TCDM)
    report.add_section(
        "L3 staging (not counted in TCDM)",
        [
            ("z (L3)", len_z),
            ("y (L3)", len_y),
        ],
    )

    report.add_peak("Phase 2 (FULL + P2 PP)", align64(full_bytes) + p2_pp)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
