#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for P1-tiled-D
#
# Phase-1-only Mamba with dInner-axis tiling; conv_out staged through L3. Phase 2 is not
# built, so iscore_out_P2 is never allocated (the FULL set is smaller than main-tiled's).
#
# FULL (always live):
#   oscore_in, iscore_out_P1 (= dt|B|C psums, BC bank-padded)
#
# Phase 1 ping-pong (×2 each):
#   oscore_weight_tile, conv_weight_tile, conv_bias_tile,
#   iscore_weight_tile, conv_out_tile (DMA to L3)
#
# Peak = align64(FULL) + P1 ping-pong

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8,
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
    dConv = derived["dConv"]

    report = MemoryReport(
        "P1-tiled-D",
        {
            "seqLen": L,
            "dModel": dModel,
            "dInner": dInner,
            "xProjDim": xProjDim,
            "nb_tiles": nb,
        },
    )

    # FULL buffers (oscore_in preloaded; iscore_out_P1 holds the dt|B|C psums)
    len_oscore_in = L * dModel * FP8 // 8
    len_iscore_out_P1 = bc_pad_iscore_out_bytes(params, L, xProjDim)  # BC bank pad §5.5

    full_bufs = [
        ("oscore_in", len_oscore_in),
        ("iscore_out_P1 (= dt_in)", len_iscore_out_P1),
    ]
    report.add_section("FULL (always live)", full_bufs)
    full_bytes = sequential_bytes([s for _, s in full_bufs])

    # Phase 1 ping-pong tiles
    len_oscore_weight = dModel * dInner * FP8 // 8
    len_conv_weight = dInner * dConv * FP8 // 8
    len_conv_bias = dInner * FP8 // 8
    len_conv_out = L * dInner * FP8 // 8
    len_iscore_weight = dInner * xProjDim * FP8 // 8

    p1_tiles = [
        ("oscore_weight_tile", len_oscore_weight // nb),
        ("conv_weight_tile", len_conv_weight // nb),
        ("conv_bias_tile", len_conv_bias // nb),
        ("iscore_weight_tile", len_iscore_weight // nb),
        ("conv_out_tile", len_conv_out // nb),
    ]
    report.add_section("Phase 1 ping-pong (×2 each)", p1_tiles)
    p1_pp = pingpong_bytes([s for _, s in p1_tiles])

    # L3 staging (not counted in TCDM)
    report.add_section(
        "L3 staging (not counted in TCDM)",
        [
            ("conv_out (L3)", len_conv_out),
        ],
    )

    report.add_peak("Phase 1 (FULL + P1 PP)", align64(full_bytes) + p1_pp)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
