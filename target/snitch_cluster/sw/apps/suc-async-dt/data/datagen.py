#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# SUC-only kernel, dInner-tiled and with the BC, x, y, and z async tiled.
# See 12_suc_async.md

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
    APP_NAME = "suc-async-dt"

    def __init__(self, **kwargs):
        DataGeneratorBase.__init__(self, self.APP_NAME, **kwargs)
        self.phase1_scalars = {}
        self.phase2_scalars = {}
        local = self.params_in_path(__file__)
        for key, value in hjson.loads(local.read_text()).items():
            self.kwargs.setdefault(key, value)

    def run(self):
        # main-tiled's tiled M2_* constants + golden data, plus this app's ring bounds + memory model.
        self.save_params()
        self.check_tiling_constraints()
        self.build_Phase1_data()
        self.build_Phase2_data()
        self._emit_suc_rings()
        self._run_memory_model()

    def _run_memory_model(self):
        # Override the parent's: it resolves the app dir from its own __file__, which would load
        # main-tiled's model + params. Use THIS app's layout (memory_model.py next to here).
        import importlib.util

        app_dir = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "memory_model_suc_async", os.path.join(app_dir, "memory_model.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from memory_model_base import run_model_from_datagen  # type: ignore[import]

        self.lines_params.append(run_model_from_datagen(mod.build_report, app_dir))

    def _emit_suc_rings(self):
        su = self.seqLenUnroll
        nb_l_tiles = self.kwargs["nb_l_tiles"]
        nb_slots = self.kwargs["nb_slots"]
        L_tile = self.seqLen // nb_l_tiles
        dInner_tile = self.dInner_tile
        assert self.bc_pad_banks % 4 == 0, "bc_pad_banks must be a multiple of 4 (else R7 lanes still alias)"
        assert L_tile % su == 0, f"L_tile {L_tile} must be a multiple of seqLenUnroll {su}"
        assert nb_l_tiles % nb_slots == 0, "nb_l_tiles must be a multiple of nb_slots"
        win_per_l_tile = L_tile // su

        # BC bank-conflict padding: pad each bank-transpose matrix to bc_matrix_stride (see 10_memsim.md).
        def pad_win(raw):  # raw spans a whole number of (unpadded) bank-transpose matrices
            assert raw % self.bc_matrix_bytes == 0, f"{raw} not a multiple of bc_matrix_bytes {self.bc_matrix_bytes}"
            return (raw // self.bc_matrix_bytes) * self.bc_matrix_stride

        dt_win = pad_win(su * self.dtRank * FP8 // 8)  # dt bytes per (padded) window (packed stride)
        bc_win = pad_win(su * (2 * self.dState) * FP8 // 8)  # BC bytes per (padded) window (ring stride)
        comb_win = dt_win + bc_win  # combined dt_BC window stride in L3 (padded)
        # The SUC re-reads BC once per delaySU channel-group. Per dInner TILE this is dInner_tile/delaySU.
        broadcast = dInner_tile // self.delaySU
        assert broadcast >= 1, f"dInner_tile {dInner_tile} must be >= delaySU {self.delaySU}"
        assert (nb_l_tiles * broadcast) % nb_slots == 0, "nb_l_tiles*broadcast must be a multiple of nb_slots"

        # dt async ring (coarser than BC): a slot spans dt_group L-tiles, so dt refills only every
        # dt_group visits (dt windows are ~5x smaller than BC, so a coarse slot stays cheap).
        nb_dt_tiles = self.kwargs["nb_dt_tiles"]
        nb_dt_slots = self.kwargs["nb_dt_slots"]
        assert nb_l_tiles % nb_dt_tiles == 0, "nb_l_tiles must be a multiple of nb_dt_tiles"
        assert nb_dt_tiles % nb_dt_slots == 0, "nb_dt_tiles must be a multiple of nb_dt_slots"
        assert (nb_dt_tiles * broadcast) % nb_dt_slots == 0, "nb_dt_tiles*broadcast must be a multiple of nb_dt_slots"
        dt_group = nb_l_tiles // nb_dt_tiles  # L-tiles per dt tile (sets the dt refill period)
        dt_slot_windows = dt_group * win_per_l_tile  # dt windows per dt slot

        # R2 (dt) ring: mirror of R7 (BC) but with dt_group-L-tile slots. Inner dims (dim0/dim1) walk
        # one window; dim2 walks the nb_dt_slots abutting slots; stride-0 dim3 wraps L-tiles + broadcast
        # (dt is re-read across the broadcast passes, exactly like BC).
        r2_tb = [
            self.dtRank * FP8 // self.switchcore_width,
            su,
            nb_dt_slots * dt_slot_windows,
            (nb_dt_tiles * broadcast) // nb_dt_slots,
        ]
        r2_ts = [pad_win((self.switchcore_width // 8) * su), BANK_BYTES, dt_win, 0]
        self.lines_params.append(f"int32_t M2_R2_tb_dt_ring[] = {{{', '.join(map(str, r2_tb))}}};")
        self.lines_params.append(f"int32_t M2_R2_ts_dt_ring[] = {{{', '.join(map(str, r2_ts))}}};")

        # R7 (BC) ring: inner dim walks nb_slots abutting slots, stride-0 wrap cycles L-tiles + broadcast.
        r7_tb = [
            (2 * self.dState * FP8) // (2 * self.suc_serial_width_BC),
            su,
            nb_slots * win_per_l_tile,
            (nb_l_tiles * broadcast) // nb_slots,
        ]
        r7_ts = [pad_win((2 * self.suc_serial_width_BC // 8) * su), BANK_BYTES, bc_win, 0]
        self.lines_params.append(f"int32_t M2_R7_tb_ring[] = {{{', '.join(map(str, r7_tb))}}};")
        self.lines_params.append(f"int32_t M2_R7_ts_ring[] = {{{', '.join(map(str, r7_ts))}}};")

        # R9 (x) / R10 (z) / W2 (y) compact per-subtile rings: one subtile-window per slot, same visit
        # schedule / R11 gauge as BC. See 12_suc_async.md.
        assert self.convUnroll == self.delaySU, "ring assumes convUnroll == delaySU"
        subtile_bytes = su * self.convUnroll * FP8 // 8
        xzy_tb = [subtile_bytes // BANK_BYTES, nb_slots * win_per_l_tile, (nb_l_tiles * broadcast) // nb_slots, 1]
        xzy_ts = [BANK_BYTES, subtile_bytes, 0, 0]
        self.lines_params.append(f"int32_t M2_xzy_tb_ring[] = {{{', '.join(map(str, xzy_tb))}}};")
        self.lines_params.append(f"int32_t M2_xzy_ts_ring[] = {{{', '.join(map(str, xzy_ts))}}};")

        # L_tile is new; main.c references it unqualified (P2 convention).
        self.lines_params.append(f"uint32_t L_tile = {L_tile};")

        for name, val in [
            ("dtBC_window_src_stride", comb_win),
            ("dt_pack_window_bytes", dt_win),
            ("dt_group", dt_group),  # L-tiles per dt slot (= dt refill every dt_group visits)
            ("dt_slot_windows", dt_slot_windows),
            ("length_dt_slot", dt_slot_windows * dt_win),
            ("BC_l3_offset", dt_win),
            ("BC_window_bytes", bc_win),
            ("BC_windows_per_l_tile", win_per_l_tile),
            ("length_BC_l_tile", win_per_l_tile * bc_win),
            ("BC_broadcast", broadcast),
            ("BC_n_visits", nb_l_tiles * broadcast),  # ring visits per dInner-tile invocation
            ("BC_gauge_step", L_tile * self.delaySU),  # R11 (SUC out) elements per BC L-tile read
            # x/z/y compact-slot sizing + L3 gather/scatter strides (share BC's visit count + gauge step).
            ("length_xzy_l_tile", win_per_l_tile * subtile_bytes),
            ("xzy_subtile_bytes", subtile_bytes),
            ("xzy_window_src_stride", su * self.dInnerUnroll * FP8 // 8),
            ("xzy_colblock_src_stride", self.seqLen * self.dInnerUnroll * FP8 // 8),
            ("xzy_subtiles_per_colblock", self.dInnerUnroll // self.convUnroll),
        ]:
            self.lines_params.append(f"uint32_t M2_{name} = {val};")


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
