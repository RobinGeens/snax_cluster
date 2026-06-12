#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `is-osgemm-tiled` app.
#
# Parallel OSGEMM + ISGEMM tiled only along dInner. There are no phases: every
# buffer is resident at once for the whole pipelined loop. The seqLen (M) axis is
# NOT tiled, so A_os and CD_is stay full-resident and grow with seqLen. The OSGEMM
# output D_os is write-only and not consumed on-chip, so each tile is spilled to L3
# and only the D_os ping-pong tile is resident. Layout: see src/main.c.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import (
    FP8, BF16, MemoryReport, pingpong_bytes, sequential_bytes, run_model,
)


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    dInner = params["dInner"]
    nb = params["nb_tiles"]

    report = MemoryReport("is-osgemm-tiled", {
        "seqLen": L, "dModel": dModel, "dInner": dInner, "nb_tiles": nb,
    })

    # FULL resident (not tiled, scale with seqLen)
    len_a_os = L * dModel * FP8 // 8        # OSGEMM A, reused by every dInner tile
    len_cd_is = L * dModel * BF16 // 8      # ISGEMM CD accumulator
    # D_os (OSGEMM out) is spilled to L3 tile-by-tile, only the ping-pong tile is resident.

    full_bufs = [
        ("A_os (osgemm A)",       len_a_os),
        ("CD_is (isgemm accum)",  len_cd_is),
    ]
    report.add_section("FULL resident (scale with seqLen)", full_bufs)
    full_bytes = sequential_bytes([s for _, s in full_bufs])

    # Ping-pong tiles (×2 each), tiled along dInner
    len_b_os_tile = (dModel * dInner * FP8 // 8) // nb
    len_d_os_tile = (L * dInner * FP8 // 8) // nb
    len_a_is_tile = (L * dInner * FP8 // 8) // nb
    len_b_is_tile = (dInner * dModel * FP8 // 8) // nb

    pp_tiles = [
        ("B_os tile", len_b_os_tile),
        ("D_os tile", len_d_os_tile),
        ("A_is tile", len_a_is_tile),
        ("B_is tile", len_b_is_tile),
    ]
    report.add_section("Ping-pong tiles (×2 each, tiled along dInner)", pp_tiles)
    pp_bytes = pingpong_bytes([s for _, s in pp_tiles])

    report.add_peak("Resident (full + ping-pong)", full_bytes + pp_bytes)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
