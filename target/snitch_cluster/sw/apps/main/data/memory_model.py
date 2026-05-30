#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `main` app (non-tiled).
# Two-phase Mamba: Phase 1 (in-proj + conv + x-proj) and Phase 2
# (z-proj + dt-proj + SUC + out-proj). All buffers live in TCDM
# sequentially; Phase 2 base starts after Phase 1's iscore_out end.
# Memory reuse: P1 conv_out = P2 x, P1 iscore_out = P2 dt_in.

import os
import sys

sys.path.append(os.path.dirname(__file__))
from memory_model_base import (
    FP8, BF16, align64, derive_mamba_params,
    MemoryReport, sequential_bytes, run_model,
    bc_pad_iscore_out_bytes, bc_pad_dt_bc_bytes,
)


def build_report(params: dict) -> MemoryReport:
    L = params["seqLen"]
    dModel = params["dModel"]
    derived = derive_mamba_params(params)
    dInner = derived["dInner"]
    xProjDim = derived["xProjDim"]
    dtRank = derived["dtRank"]
    dConv = derived["dConv"]
    dState = derived["dState"]
    dtRankUnroll = 6

    report = MemoryReport("main", {
        "seqLen": L, "dModel": dModel, "dInner": dInner,
        "xProjDim": xProjDim, "dtRank": dtRank,
    })

    # Phase 1 buffers (sequential in TCDM)
    p1_bufs = [
        ("oscore_in",      L * dModel * FP8 // 8),
        ("oscore_weight",  dModel * dInner * FP8 // 8),
        ("conv_weight",    dInner * dConv * FP8 // 8),
        ("conv_bias",      dInner * FP8 // 8),
        ("conv_out",       L * dInner * FP8 // 8),
        ("iscore_weight",  dInner * xProjDim * FP8 // 8),
        ("iscore_out",     bc_pad_iscore_out_bytes(params, L, xProjDim)),
    ]
    report.add_section("Phase 1 (sequential in TCDM)", p1_bufs)
    p1_total = sequential_bytes([s for _, s in p1_bufs])

    # Phase 2 buffers (sequential, base = after P1 iscore_out)
    p2_bufs = [
        ("oscore_in",      L * dModel * FP8 // 8),
        ("oscore_weight",  dModel * dInner * FP8 // 8),
        ("z",              L * dInner * FP8 // 8),
        ("dt_BC",          bc_pad_dt_bc_bytes(params, L, xProjDim)),
        ("dt_weight_1",    dInner * (dtRank // dtRankUnroll) * dConv * FP8 // 8),
        ("dt_weight_2",    dInner * (dtRank // dtRankUnroll) * (dtRankUnroll - dConv) * FP8 // 8),
        ("dt_bias",        dInner * FP8 // 8),
        ("x",              L * dInner * FP8 // 8),
        ("A",              dInner * dState * FP8 // 8),
        ("D",              dInner * FP8 // 8),
        ("y",              L * dInner * FP8 // 8),
        ("iscore_weight",  dModel * dInner * FP8 // 8),
        ("iscore_out",     L * dModel * BF16 // 8),
    ]
    report.add_section("Phase 2 (sequential, starts after P1 iscore_out)", p2_bufs)
    p2_total = sequential_bytes([s for _, s in p2_bufs])

    # Peak: P1's iscore_out end + P2's max endpoint
    # In main.c, P2 base = P1.addr_iscore_out + P1.length_iscore_out
    p1_iscore_out_end = sequential_bytes([s for _, s in p1_bufs])
    fused_peak = p1_iscore_out_end + p2_total

    report.add_peak("Phase 1 alone", p1_total)
    report.add_peak("Phase 2 alone", p2_total)
    report.add_peak("Fused P1+P2", fused_peak)
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
