#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Always live:
#   oscore_in ring (nb_slots slots of L_tile*dModel), dt_BC (FULL)
#
# Phase 2 ping-pong (×2 each):
#   oscore_weight_tile, dt_weight_1_tile, dt_weight_2_tile, dt_bias_tile,
#   A_tile, D_tile, x_tile (DMA from L3), z_tile (DMA to L3), y_tile (DMA to L3)
#
# Peak = align64(ring + dt_BC) + P2_pingpong

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
    bc_pad_dt_bc_bytes,
    run_model,
)


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    nb = params["nb_tiles"]
    nb_l_tiles = params["nb_l_tiles"]
    nb_slots = params["nb_slots"]
    derived = derive_mamba_params(params)
    dInner = derived["dInner"]
    xProjDim = derived["xProjDim"]
    dtRank = derived["dtRank"]
    dConv = derived["dConv"]
    dState = derived["dState"]
    dtRankUnroll = 6

    report = MemoryReport(
        "P2-async-OS-no-IS",
        {
            "seqLen": L,
            "dModel": dModel,
            "dInner": dInner,
            "nb_tiles": nb,
            "nb_l_tiles": nb_l_tiles,
            "nb_slots": nb_slots,
        },
    )

    # Always-live buffers: oscore_in ring (nb_slots L-tiles) + PACKED full dt + BC ring (nb_slots
    # L-tiles, refilled from the combined dt_BC L3 buffer). dt stays full (small); BC rings.
    L_tile = L // nb_l_tiles
    len_oscore_in_l_tile = L_tile * dModel * FP8 // 8
    len_oscore_in_ring = nb_slots * len_oscore_in_l_tile
    len_dt_packed = L * dtRank * FP8 // 8
    len_bc_l_tile = L_tile * (2 * dState) * FP8 // 8
    len_bc_ring = nb_slots * len_bc_l_tile

    live_bufs = [
        (f"oscore_in ring ({nb_slots}x{len_oscore_in_l_tile}B)", len_oscore_in_ring),
        ("dt PACKED (full)", len_dt_packed),
        (f"BC ring ({nb_slots}x{len_bc_l_tile}B)", len_bc_ring),
    ]
    report.add_section("Always live", live_bufs)
    live_bytes = sequential_bytes([s for _, s in live_bufs])

    # Phase 2 ping-pong tiles (no iscore_weight / iscore_out: IS-core excluded).
    len_oscore_weight = dModel * dInner * FP8 // 8
    len_dt_w1 = dInner * (dtRank // dtRankUnroll) * dConv * FP8 // 8
    len_dt_w2 = dInner * (dtRank // dtRankUnroll) * (dtRankUnroll - dConv) * FP8 // 8
    len_dt_bias = dInner * FP8 // 8
    len_A = dInner * dState * FP8 // 8
    len_D = dInner * FP8 // 8
    len_x = L * dInner * FP8 // 8
    len_z = L * dInner * FP8 // 8
    len_y = L * dInner * FP8 // 8

    # Per-dInner-tile SINGLE buffers (×1): the loop is serialized so x/z/y/weights are not ping-ponged.
    p2_tiles = [
        ("oscore_weight", len_oscore_weight // nb),
        ("dt_weight_1", len_dt_w1 // nb),
        ("dt_weight_2", len_dt_w2 // nb),
        ("dt_bias", len_dt_bias // nb),
        ("A", len_A // nb),
        ("D", len_D // nb),
        ("x (FULL-L)", len_x // nb),
        ("z (FULL-L)", len_z // nb),
        ("y (FULL-L)", len_y // nb),
    ]
    report.add_section("Phase 2 single-buffer (×1 each)", p2_tiles)
    p2_pp = sequential_bytes([s for _, s in p2_tiles])

    # L3 staging (not counted in TCDM)
    l3_bufs = [
        ("z (L3)", len_z),
        ("y (L3)", len_y),
    ]
    report.add_section("L3 staging (not counted in TCDM)", l3_bufs)

    report.add_peak("Phase 2 (live + P2 PP)", align64(live_bytes) + p2_pp)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
