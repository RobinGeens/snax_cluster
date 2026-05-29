#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

import math
import logging
import os
import sys
import hjson

# ── HW constants (must match snax_simbacore_cluster.hjson / RTL) ──────────

TCDM_BYTES = 512 * 1024
FP8 = 8
BF16 = 16
BANKWIDTH = 64
BANK_BYTES = BANKWIDTH // 8

# Unroll factors / widths (fixed in RTL)
SEQ_LEN_UNROLL = 16
D_INNER_UNROLL = 24
DT_RANK_UNROLL = 6
CONV_UNROLL = 4
DELAY_SU = 4
SIMD_LANES_BF16 = 16
SIMD_LANES_FP8 = 32

# Hardwired algorithmic constants
D_STATE = 64
D_CONV = 4
DT_RANK = 24
EXPAND = 2

logger = logging.getLogger(__name__)


# ── Utilities ─────────────────────────────────────────────────────────────

def align64(v: int) -> int:
    return (v + 63) & ~63


def pad_to_unroll(dim: int, unroll: int = D_INNER_UNROLL) -> int:
    return math.ceil(dim / unroll) * unroll


def derive_mamba_params(user: dict) -> dict:
    """Derive dInner, xProjDim etc. from user-specified params."""
    dModel = user["dModel"]
    dtRank = user.get("dtRank", DT_RANK)
    dInner = EXPAND * dModel
    xProjDim = dtRank + 2 * D_STATE
    return {
        "dInner": dInner,
        "xProjDim": xProjDim,
        "dState": D_STATE,
        "dConv": D_CONV,
        "dtRank": dtRank,
    }


def read_params_in(hjson_path: str) -> dict:
    with open(hjson_path) as f:
        return hjson.loads(f.read())


# ── Report builder ────────────────────────────────────────────────────────

class MemoryReport:
    def __init__(self, app_name: str, params: dict):
        self.app_name = app_name
        self.params = params
        self.sections: list[tuple[str, list[tuple[str, int]]]] = []
        self.phase_peaks: list[tuple[str, int]] = []

    def add_section(self, title: str, buffers: list[tuple[str, int]]):
        self.sections.append((title, buffers))

    def add_peak(self, phase_name: str, peak_bytes: int):
        self.phase_peaks.append((phase_name, peak_bytes))

    def overall_peak(self) -> int:
        return max(p for _, p in self.phase_peaks) if self.phase_peaks else 0

    @staticmethod
    def _kib(b: int) -> str:
        return f"{b / 1024:.2f} KiB"

    def format(self) -> str:
        lines: list[str] = []
        peak = self.overall_peak()
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        lines.append(f"{'=' * 60}")
        lines.append(f"  Memory Model: {self.app_name}")
        lines.append(f"  Parameters: {param_str}")
        lines.append(f"{'=' * 60}")
        lines.append("")

        for title, buffers in self.sections:
            lines.append(f"--- {title} ---")
            subtotal = 0
            for name, size in buffers:
                lines.append(f"  {name:<40s} {self._kib(size):>12s}")
                subtotal += size
            lines.append(f"  {'Subtotal':<40s} {self._kib(subtotal):>12s}")
            lines.append("")

        lines.append(f"--- Peak TCDM Usage ---")
        for phase_name, peak_bytes in self.phase_peaks:
            flag = " *** OOM ***" if peak_bytes > TCDM_BYTES else ""
            lines.append(f"  {phase_name:<40s} {self._kib(peak_bytes):>12s}{flag}")
        lines.append(f"  {'Overall peak':<40s} {self._kib(peak):>12s}")
        lines.append(f"  {'Budget':<40s} {self._kib(TCDM_BYTES):>12s}")
        if peak > TCDM_BYTES:
            lines.append(f"\n  *** OOM: exceeds budget by {self._kib(peak - TCDM_BYTES)} ***")
        else:
            lines.append(f"\n  OK ({self._kib(TCDM_BYTES - peak)} headroom)")
        lines.append("")
        return "\n".join(lines)

    def write(self, filepath: str):
        text = self.format()
        with open(filepath, "w") as f:
            f.write(text)

    def check_oom(self):
        peak = self.overall_peak()
        if peak > TCDM_BYTES:
            print(self.format(), file=sys.stderr)
            logger.warning(
                "%s: L1 footprint %s exceeds TCDM budget %s.",
                self.app_name,
                self._kib(peak),
                self._kib(TCDM_BYTES),
            )


def pingpong_bytes(tile_lengths: list[int]) -> int:
    """Bytes consumed by two aligned ping-pong slots per tiled tensor."""
    cursor = 0
    for length in tile_lengths:
        cursor = align64(cursor + length)
        cursor = align64(cursor + length)
    return cursor


def sequential_bytes(lengths: list[int]) -> int:
    """Bytes consumed by sequentially aligned buffers."""
    cursor = 0
    for length in lengths:
        cursor = align64(cursor + length)
    return cursor


def run_model(model_fn, app_dir: str):
    """Entry point for standalone execution. Reads params_in.hjson from app_dir,
    runs the model, writes the report, and checks OOM."""
    params_path = os.path.join(app_dir, "params_in.hjson")
    params = read_params_in(params_path)
    report = model_fn(params)
    report_path = os.path.join(app_dir, "memory_report.txt")
    report.write(report_path)
    print(report.format())
    report.check_oom()
    return report


def run_model_from_datagen(model_fn, app_dir: str):
    """Called from datagen to run the memory model, write report, and check OOM.
    Returns C source lines (comment + L1_TCDM_PEAK_BYTES define) for data.h."""
    params_path = os.path.join(app_dir, "params_in.hjson")
    params = read_params_in(params_path)
    report = model_fn(params)
    report_path = os.path.join(app_dir, "memory_report.txt")
    report.write(report_path)
    report.check_oom()
    peak = report.overall_peak()
    return (
        f"// L1 TCDM peak: {peak / 1024:.2f} KiB — see memory_report.txt\n"
        f"#define L1_TCDM_PEAK_BYTES {peak}u"
    )
