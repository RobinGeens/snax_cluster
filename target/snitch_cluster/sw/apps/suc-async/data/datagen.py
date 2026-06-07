#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# suc-async = SUC-only kernel, dInner-tiled (like SUC-tiled) AND with the BC input async L-tiled into
# a TCDM ring (like the original suc-async). Reuses main-tiled's Phase-2 tiled constants (per-dInner
# M2_*_tile sizes + the FULL combined dt_BC buffer) and additionally emits the packed-dt / BC-ring
# streamer bounds, sized for one dInner tile. Ring mechanics: docs/dataflow/09_async_tiling.md.


import os
import sys
import importlib.util

import hjson

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "../../main/data"))

from datagen_base import BANK_BYTES, FP8, DataGeneratorBase  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]

# Load main-tiled's DataGenerator by file path (the local module is also named datagen.py, so a plain
# import would re-import THIS module -> circular import).
_spec = importlib.util.spec_from_file_location(
    "_main_tiled_datagen", os.path.abspath(os.path.join(_HERE, "../../main-tiled/data/datagen.py"))
)
_main_tiled_datagen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main_tiled_datagen)


class DataGenerator(_main_tiled_datagen.DataGenerator):
    APP_NAME = "suc-async"

    def __init__(self, **kwargs):
        DataGeneratorBase.__init__(self, self.APP_NAME, **kwargs)
        self.phase1_scalars = {}
        self.phase2_scalars = {}
        local = self.params_in_path(__file__)
        for key, value in hjson.loads(local.read_text()).items():
            self.kwargs.setdefault(key, value)

    def run(self):
        # main-tiled's tiled M2_* constants + golden data (skip the memory model: no memory_model.py here).
        self.save_params()
        self.check_tiling_constraints()
        self.build_Phase1_data()
        self.build_Phase2_data()
        self._emit_suc_rings()

    def _emit_suc_rings(self):
        su = self.seqLenUnroll
        nb_l_tiles = self.kwargs["nb_l_tiles"]
        nb_slots = self.kwargs["nb_slots"]
        L_tile = self.seqLen // nb_l_tiles
        dInner_tile = self.dInner_tile
        assert self.bc_pad_bytes == 0, "suc-async ring assumes non-padded dt_BC (set bc_pad_banks=0)"
        assert L_tile % su == 0, f"L_tile {L_tile} must be a multiple of seqLenUnroll {su}"
        assert nb_l_tiles % nb_slots == 0, "nb_l_tiles must be a multiple of nb_slots"
        win_per_l_tile = L_tile // su

        dt_win = su * self.dtRank * FP8 // 8  # dt bytes per window (packed stride)
        bc_win = su * (2 * self.dState) * FP8 // 8  # BC bytes per window (ring stride)
        comb_win = su * self.xProjDim * FP8 // 8  # combined dt_BC window stride in L3
        # The SUC re-reads BC once per delaySU channel-group. Per dInner TILE this is dInner_tile/delaySU.
        broadcast = dInner_tile // self.delaySU
        assert broadcast >= 1, f"dInner_tile {dInner_tile} must be >= delaySU {self.delaySU}"
        assert (nb_l_tiles * broadcast) % nb_slots == 0, "nb_l_tiles*broadcast must be a multiple of nb_slots"

        # R2 (dt) PACKED, sized for one dInner tile: base R2 temporal bounds, L-window stride shrunk
        # comb_win -> dt_win, outermost (channel-group) bound = dInner_tile/convUnroll.
        r2_tb = [self.dtRank * FP8 // self.switchcore_width, su, self.seqLen // su, dInner_tile // self.convUnroll]
        r2_ts = [(self.switchcore_width // 8) * su, BANK_BYTES, dt_win, 0]
        self.lines_params.append(f"int32_t M2_R2_tb_packed[] = {{{', '.join(map(str, r2_tb))}}};")
        self.lines_params.append(f"int32_t M2_R2_ts_packed[] = {{{', '.join(map(str, r2_ts))}}};")

        # R7 (BC) RING, sized for one dInner tile: inner dim walks nb_slots abutting slots; the stride-0
        # wrap absorbs the L-tile cycle and the per-tile broadcast re-reads.
        r7_tb = [
            (2 * self.dState * FP8) // (2 * self.suc_serial_width_BC),
            su,
            nb_slots * win_per_l_tile,
            (nb_l_tiles * broadcast) // nb_slots,
        ]
        r7_ts = [(2 * self.suc_serial_width_BC // 8) * su, BANK_BYTES, bc_win, 0]
        self.lines_params.append(f"int32_t M2_R7_tb_ring[] = {{{', '.join(map(str, r7_tb))}}};")
        self.lines_params.append(f"int32_t M2_R7_ts_ring[] = {{{', '.join(map(str, r7_ts))}}};")

        # nb_l_tiles / nb_slots / nb_tiles are already emitted by the base (params). Only L_tile is new;
        # main.c references it UNqualified (P2 convention).
        self.lines_params.append(f"uint32_t L_tile = {L_tile};")

        for name, val in [
            ("dtBC_window_src_stride", comb_win),
            ("dt_pack_window_bytes", dt_win),
            ("dt_windows_total", self.seqLen // su),
            ("length_dt_packed", (self.seqLen // su) * dt_win),
            ("BC_l3_offset", dt_win),
            ("BC_window_bytes", bc_win),
            ("BC_windows_per_l_tile", win_per_l_tile),
            ("length_BC_l_tile", win_per_l_tile * bc_win),
            ("BC_broadcast", broadcast),
            ("BC_n_visits", nb_l_tiles * broadcast),  # ring visits per dInner-tile invocation
            ("BC_gauge_step", L_tile * self.delaySU),  # R11 (SUC out) elements per BC L-tile read
        ]:
            self.lines_params.append(f"uint32_t M2_{name} = {val};")


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
