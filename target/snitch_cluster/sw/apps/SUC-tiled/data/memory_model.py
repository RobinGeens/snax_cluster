#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `SUC-tiled` app.
#
# Synchronous SU-core-only Phase 2 tiled along dInner. dt+BC is loaded once and stays
# full-resident; every other buffer (dt weights/bias, A, D, x, z, y) is a dInner tile,
# double-buffered ping-pong. Sizes mirror main-tiled's P2 tiled constants. Layout:
#   [ dt_BC (full) | dt_w1(2) | dt_w2(2) | dt_bias(2) | A(2) | D(2) | x(2) | z(2) | y(2) ]
# The full y output is spilled to L3 per tile (not counted in TCDM). See src/main.c.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (  # type: ignore[import]
    FP8, DT_RANK_UNROLL, align64, derive_mamba_params, bc_pad_dt_bc_bytes,
    MemoryReport, pingpong_bytes, run_model,
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
    dtRankUnroll = DT_RANK_UNROLL

    report = MemoryReport("SUC-tiled", {
        "seqLen": L, "dModel": dModel, "dInner": dInner, "nb_tiles": nb,
    })

    # Full dt+BC (not tiled), incl. BC bank-conflict padding (§5.5)
    len_dt_BC = bc_pad_dt_bc_bytes(params, L, xProjDim)
    report.add_section("Full resident", [("dt_BC (dt + BC, full)", len_dt_BC)])

    # Per-dInner-tile ping-pong (x2 each)
    len_dt_w1 = dInner * (dtRank // dtRankUnroll) * dConv * FP8 // 8
    len_dt_w2 = dInner * (dtRank // dtRankUnroll) * (dtRankUnroll - dConv) * FP8 // 8
    len_dt_bias = dInner * FP8 // 8
    len_A = dInner * dState * FP8 // 8
    len_D = dInner * FP8 // 8
    len_xzy = L * dInner * FP8 // 8

    pp_tiles = [
        ("dt_weight_1_tile", len_dt_w1 // nb),
        ("dt_weight_2_tile", len_dt_w2 // nb),
        ("dt_bias_tile",     len_dt_bias // nb),
        ("A_tile",           len_A // nb),
        ("D_tile",           len_D // nb),
        ("x_tile",           len_xzy // nb),
        ("z_tile",           len_xzy // nb),
        ("y_tile",           len_xzy // nb),
    ]
    report.add_section("Ping-pong (x2 each, tiled along dInner)", pp_tiles)

    report.add_section("L3 staging (not counted in TCDM)", [("y (L3, full)", len_xzy)])

    peak = align64(len_dt_BC) + pingpong_bytes([s for _, s in pp_tiles])
    report.add_peak("SUC-tiled (dt_BC full + ping-pong)", peak)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
