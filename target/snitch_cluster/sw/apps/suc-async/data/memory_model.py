#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# suc-async TCDM layout (see src/main.c). SUC-only: no osCore, so no oscore_in ring / oscore_weight
# (cf. P2-async-OS-no-IS). All buffers are laid out in ONE sequentially-64B-aligned chain and are
# live together during the per-dInner-tile compute loop, so the peak is just their sum:
#
#   dt PACKED (full L) -> BC ring (nb_slots) -> dt_w1 -> dt_w2 -> dt_bias -> A -> D
#       -> x ring (nb_slots) -> z ring (nb_slots) -> y ring (nb_slots)
#
# BC, x, z (in) and y (out) are all L-tiled into nb_slots-slot async rings, so only dt PACKED stays
# full-seqLen-resident. This keeps the TCDM footprint flat in seqLen (the point of the rings).

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8,
    BANK_BYTES,
    SEQ_LEN_UNROLL,
    CONV_UNROLL,
    derive_mamba_params,
    MemoryReport,
    sequential_bytes,
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
    dtRank = derived["dtRank"]
    dConv = derived["dConv"]
    dState = derived["dState"]
    dtRankUnroll = 6

    report = MemoryReport(
        "suc-async",
        {
            "seqLen": L,
            "dModel": dModel,
            "dInner": dInner,
            "nb_tiles": nb,
            "nb_l_tiles": nb_l_tiles,
            "nb_slots": nb_slots,
        },
    )

    L_tile = L // nb_l_tiles

    # Shared across all dInner tiles: PACKED full dt + BC nb_slots-slot async ring (refilled from the
    # combined dt_BC L3 buffer). Both stay resident for the whole run.
    # dt + BC are padded per bank-transpose matrix (bc_pad_banks, see datagen.py / 10_memsim.md).
    bc_matrix_bytes = SEQ_LEN_UNROLL * BANK_BYTES
    bc_matrix_stride = bc_matrix_bytes + params.get("bc_pad_banks", 0) * BANK_BYTES
    pad_win = lambda raw: (raw // bc_matrix_bytes) * bc_matrix_stride
    len_dt_packed = pad_win(L * dtRank * FP8 // 8)
    len_bc_l_tile = pad_win(L_tile * (2 * dState) * FP8 // 8)
    len_bc_ring = nb_slots * len_bc_l_tile

    # Per-dInner-tile single buffers (reloaded each invocation, ×1 — the loop is serialized). dt
    # weights/bias, A, D are tile-sized; sizes mirror main-tiled's P2 tiled constants.
    len_dt_w1 = (dInner * (dtRank // dtRankUnroll) * dConv * FP8 // 8) // nb
    len_dt_w2 = (dInner * (dtRank // dtRankUnroll) * (dtRankUnroll - dConv) * FP8 // 8) // nb
    len_dt_bias = (dInner * FP8 // 8) // nb
    len_A = (dInner * dState * FP8 // 8) // nb
    len_D = (dInner * FP8 // 8) // nb

    # x/z (in) and y (out) are now nb_slots-slot async rings of compact per-subtile slots, not full-L
    # buffers. A slot holds win_per_l_tile subtile-windows (seqLenUnroll x convUnroll FP8 each).
    subtile_bytes = SEQ_LEN_UNROLL * CONV_UNROLL * FP8 // 8
    win_per_l_tile = (L // nb_l_tiles) // SEQ_LEN_UNROLL
    len_xzy_ring = nb_slots * win_per_l_tile * subtile_bytes

    # One sequentially-aligned chain, in src/main.c pointer order — all live during compute.
    live_bufs = [
        ("dt PACKED (full)", len_dt_packed),
        (f"BC ring ({nb_slots}x{len_bc_l_tile}B)", len_bc_ring),
        ("dt_weight_1", len_dt_w1),
        ("dt_weight_2", len_dt_w2),
        ("dt_bias", len_dt_bias),
        ("A", len_A),
        ("D", len_D),
        (f"x ring ({nb_slots} slots)", len_xzy_ring),
        (f"z ring ({nb_slots} slots)", len_xzy_ring),
        (f"y ring ({nb_slots} slots)", len_xzy_ring),
    ]
    report.add_section("Resident during compute (×1 each)", live_bufs)

    # L3 staging (not counted in TCDM): the full y output is spilled to L3 per dInner tile.
    report.add_section("L3 staging (not counted in TCDM)", [("y (L3, full)", L * dInner * FP8 // 8)])

    report.add_peak("SUC compute (all buffers live)", sequential_bytes([s for _, s in live_bufs]))
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
