#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Data + per-tile streamer bounds for the suc-carry app. See docs/dataflow/13_suc_carry.md.

import os
import sys
import importlib.util

import hjson

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "../../main/data"))

from datagen_base import BANK_BYTES, BF16, FP8, NUM_LOOPS, DataGeneratorBase  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]

# Reuse main-tiled's DataGenerator (Phase1/2 data + default M2_* streamer bounds); loaded by path to avoid
# the name clash with this module.
_spec = importlib.util.spec_from_file_location(
    "_main_tiled_datagen", os.path.abspath(os.path.join(_HERE, "../../main-tiled/data/datagen.py"))
)
assert _spec is not None and _spec.loader is not None
_main_tiled_datagen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main_tiled_datagen)


class DataGenerator(_main_tiled_datagen.DataGenerator):
    APP_NAME = "suc-carry"

    def __init__(self, **kwargs):
        DataGeneratorBase.__init__(self, self.APP_NAME, **kwargs)
        self.phase1_scalars = {}
        self.phase2_scalars = {}
        local = self.params_in_path(__file__)
        for key, value in hjson.loads(local.read_text()).items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.save_params()
        self.check_tiling_constraints()
        self.build_Phase1_data()
        self.build_Phase2_data()
        self._emit_suc_carry()

    def _pad_win(self, raw):
        assert raw % self.bc_matrix_bytes == 0, f"{raw} not a multiple of bc_matrix_bytes {self.bc_matrix_bytes}"
        return (raw // self.bc_matrix_bytes) * self.bc_matrix_stride

    def _emit_suc_carry(self):
        assert self.nb_tiles == 1, "runs full dInner per invocation (nb_tiles must be 1)"
        su = self.seqLenUnroll
        dI = self.dInner
        nb_l_tiles = self.kwargs["nb_l_tiles"]
        L_tile = self.seqLen // nb_l_tiles
        assert self.seqLen % nb_l_tiles == 0 and L_tile % su == 0
        win_per_l_tile = L_tile // su
        broadcast = dI // self.delaySU
        assert self.convUnroll == self.delaySU

        dt_win = self._pad_win(su * self.dtRank * FP8 // 8)
        bc_win = self._pad_win(su * (2 * self.dState) * FP8 // 8)
        comb_win = dt_win + bc_win
        subtile_bytes = su * self.convUnroll * FP8 // 8
        len_xzy_l_tile = win_per_l_tile * subtile_bytes

        emit = self.lines_params.append

        # ---- compact per-tile geometry (main.c drives the per-tile DMA with these) ----
        for name, val in [
            ("L_tile", L_tile),
            ("NB_L_TILES", nb_l_tiles),
            ("BC_broadcast", broadcast),
            ("BC_windows_per_l_tile", win_per_l_tile),
            ("dtBC_window_src_stride", comb_win),
            ("dt_pack_window_bytes", dt_win),
            ("length_dt_l_tile", win_per_l_tile * dt_win),
            ("BC_l3_offset", dt_win),
            ("BC_window_bytes", bc_win),
            ("length_BC_l_tile", win_per_l_tile * bc_win),
            ("length_xzy_l_tile", len_xzy_l_tile),
            ("xzy_subtile_bytes", subtile_bytes),
            ("xzy_window_src_stride", su * self.dInnerUnroll * FP8 // 8),
            ("xzy_colblock_src_stride", self.seqLen * self.dInnerUnroll * FP8 // 8),
            ("xzy_subtiles_per_colblock", self.dInnerUnroll // self.convUnroll),
            ("xzy_n_colblocks", broadcast // (self.dInnerUnroll // self.convUnroll)),
            ("xzy_colblock_bytes", win_per_l_tile * su * self.dInnerUnroll * FP8 // 8),
        ]:
            emit(f"uint32_t M2_{name} = {val};")

        # ---- non-ring per-tile temporal bounds (strides/spatial reuse the default M2_*_ts / M2_*_ss) ----
        # dt (R2): one L-tile of the resident unpacked-dt buffer; re-read across dInner groups (stride 0).
        r2 = [self.dtRank * FP8 // self.switchcore_width, su, win_per_l_tile, dI // self.convUnroll]
        r2_ts = [self._pad_win((self.switchcore_width // 8) * su), BANK_BYTES, dt_win, 0]
        # BC (R7): resident L-tile, re-read `broadcast` times (stride 0).
        r7 = [(2 * self.dState * FP8) // (2 * self.suc_serial_width_BC), su, win_per_l_tile, broadcast]
        r7_ts = [self._pad_win((2 * self.suc_serial_width_BC // 8) * su), BANK_BYTES, bc_win, 0]
        # x/z/y (R9/R10/W2): buffer holds one L-tile in ConvFormat ([col-block][window][subtile]), gathered
        # contiguously per col-block. The streamer transposes ConvFormat -> SUCFormat here (col-block-major,
        # subtile-within-colblock, then windows), so the DMA stays one contiguous burst per col-block.
        spc = self.dInnerUnroll // self.convUnroll  # subtiles per col-block
        n_cb = broadcast // spc                      # col-blocks
        window_bytes = spc * subtile_bytes           # = su * dInnerUnroll * FP8 // 8
        xzy = [subtile_bytes // BANK_BYTES, win_per_l_tile, spc, n_cb]
        xzy_ts = [BANK_BYTES, window_bytes, subtile_bytes, win_per_l_tile * window_bytes]
        for name, tb, ts in [("R2", r2, r2_ts), ("R7", r7, r7_ts),
                             ("R9", xzy, xzy_ts), ("R10", xzy, xzy_ts), ("W2", xzy, xzy_ts)]:
            tb = list(tb) + [1] * (NUM_LOOPS - len(tb))
            ts = list(ts) + [0] * (NUM_LOOPS - len(ts))
            emit(f"int32_t M2_{name}_tb_sync[] = {{{', '.join(map(str, tb))}}};")
            emit(f"int32_t M2_{name}_ts_sync[] = {{{', '.join(map(str, ts))}}};")

        # ---- state carry through isCore ports (R13 in / W3 out): linear over dInner state vectors ----
        state_beat_bytes = su * BF16 // 8
        state_beats = dI * self.dState * BF16 // (state_beat_bytes * 8)
        emit(f"uint32_t M2_length_state_buf = {dI * self.dState * BF16 // 8};")
        for name in ("R13", "W3"):
            emit(f"int32_t M2_{name}_state_tb[] = {{{state_beats}, 1, 1, 1}};")
            emit(f"int32_t M2_{name}_state_ts[] = {{{state_beat_bytes}, 0, 0, 0}};")
            emit(f"int32_t M2_{name}_state_ss[] = {{{BANK_BYTES}}};")


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
