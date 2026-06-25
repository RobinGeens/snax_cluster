#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `double-gemm-conflict-free` app.
#
# Same parallel OSGEMM + ISGEMM as is-osgemm-tiled, but bank-partitioned with the skip-128
# layou. OSGEMM lives in banks 0-15 (low half of every block), ISGEMM in banks 16-31 (high half).
# The two heaps lverlap the same address region -- OS fills the low halves, IS the high halves of the same blocks
# Each buffer still costs 2x its logical size, but the total footprint is 2*max(OS, IS) instead of 2*OS + 2*IS.
# See docs/dataflow/20_double_gemm_conflict_free.md

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8,
    BF16,
    MemoryReport,
    run_model,
)

SKIP = 2  # skip-128 layout doubles each buffer's address footprint


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    dInner = params["dInner"]
    nb = params["nb_tiles"]

    report = MemoryReport(
        "double-gemm-conflict-free",
        {
            "seqLen": L,
            "dModel": dModel,
            "dInner": dInner,
            "nb_tiles": nb,
        },
    )

    # OSGEMM heap (low halves, banks 0-15). A_os full resident; B_os/D_os ping-ponged (x2).
    len_a_os = SKIP * (L * dModel * FP8 // 8)  # reused by every dInner tile
    len_b_os_tile = SKIP * ((dModel * dInner * FP8 // 8) // nb)
    len_d_os_tile = SKIP * ((L * dInner * FP8 // 8) // nb)  # spilled to L3, ping-pong resident
    os_bufs = [
        ("A_os (osgemm A, full)", len_a_os),
        ("B_os tile (x2)", 2 * len_b_os_tile),
        ("D_os tile (x2)", 2 * len_d_os_tile),
    ]
    report.add_section("OSGEMM heap -> banks 0-15 (low half)", os_bufs)
    os_span = sum(s for _, s in os_bufs)

    # ISGEMM heap (high halves, banks 16-31). CD_is full resident; A_is/B_is ping-ponged (x2).
    len_cd_is = SKIP * (L * dModel * BF16 // 8)  # accumulator
    len_a_is_tile = SKIP * ((L * dInner * FP8 // 8) // nb)
    len_b_is_tile = SKIP * ((dInner * dModel * FP8 // 8) // nb)
    is_bufs = [
        ("CD_is (isgemm accum, full)", len_cd_is),
        ("A_is tile (x2)", 2 * len_a_is_tile),
        ("B_is tile (x2)", 2 * len_b_is_tile),
    ]
    report.add_section("ISGEMM heap -> banks 16-31 (high half)", is_bufs)
    is_span = sum(s for _, s in is_bufs)

    # The two heaps overlap the same region; peak is the larger side, not the sum.
    report.add_peak("Resident (2*max(OS, IS) overlap)", max(os_span, is_span))
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
