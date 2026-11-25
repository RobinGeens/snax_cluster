#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Standalone L1 TCDM memory model for the `osgemm-tiled` app. See docs/dataflow/01_oscore_kernels.md.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
from memory_model_base import FP8, MemoryReport, sequential_bytes, run_model  # type: ignore[import]

MU, NU = 16, 24  # OS-core seqLenUnroll, dInnerUnroll


def split(total, n):
    assert 1 <= n <= total, f"cannot split {total} blocks into {n} parts"
    return [total // n + (i < total % n) for i in range(n)]


def plan(params: dict) -> dict:
    M, K, N = params["dim0"], params["dim1"], params["dim2"]
    assert M % MU == 0, f"dim0 ({M}) must be a multiple of {MU}"
    assert N % NU == 0, f"dim2 ({N}) must be a multiple of {NU}"
    assert K % 4 == 0, f"dim1 ({K}) must be a multiple of 4 (weight downsizer)"
    chunks = split(M // MU, params["nb_l_tiles"])  # M-blocks per activation chunk
    tiles = split(N // NU, params["nb_tiles"])  # N-blocks per weight tile
    a_stat = bool(params["a_stationary"])
    a_chunk = max(chunks) * MU * K * FP8 // 8
    b_tile = K * max(tiles) * NU * FP8 // 8
    d_tile = max(chunks) * MU * max(tiles) * NU * FP8 // 8
    outer, inner = (a_chunk, b_tile) if a_stat else (b_tile, a_chunk)
    n_inner = len(tiles) if a_stat else len(chunks)
    launches = len(chunks) * len(tiles)
    assert n_inner > 1 or launches == 1, (
        f"a_stationary={int(a_stat)} streams a single {'weight tile' if a_stat else 'activation chunk'}: "
        f"use a_stationary={int(not a_stat)}")
    return dict(
        chunks=chunks,
        tiles=tiles,
        a_stationary=a_stat,
        len_a_chunk=a_chunk,
        len_b_tile=b_tile,
        len_d_tile=d_tile,
        len_outer=outer,
        len_inner=inner,
        nb_inner_bufs=2 if n_inner > 1 else 1,
        nb_d_bufs=2 if launches > 1 else 1,
    )


def dram_bytes(params: dict) -> dict:
    """L3 DMA bytes of one call"""
    p = plan(params)
    M, K, N = params["dim0"], params["dim1"], params["dim2"]
    a, b, d = M * K, K * N, M * N
    if p["a_stationary"]:
        b *= len(p["chunks"]) if len(p["tiles"]) > 1 else 1
    else:
        a *= len(p["tiles"]) if len(p["chunks"]) > 1 else 1
    return dict(read=a + b, write=d)


def build_report(params: dict) -> MemoryReport:
    p = plan(params)
    report = MemoryReport("osgemm-tiled", {k: params[k] for k in
                                           ("dim0", "dim1", "dim2", "nb_tiles", "nb_l_tiles", "a_stationary")})
    res, stream = ("A chunk", "B tile") if p["a_stationary"] else ("B tile", "A chunk")
    bufs = [(f"{res} (resident)", p["len_outer"])]
    bufs += [(f"{stream} {i} (streamed)", p["len_inner"]) for i in range(p["nb_inner_bufs"])]
    bufs += [(f"D tile {i} (to L3)", p["len_d_tile"]) for i in range(p["nb_d_bufs"])]
    report.add_section("Buffers (largest chunk / tile)", bufs)
    report.add_peak("Resident", sequential_bytes([s for _, s in bufs]))
    return report


if __name__ == "__main__":
    run_model(build_report, os.path.dirname(os.path.abspath(__file__)))
