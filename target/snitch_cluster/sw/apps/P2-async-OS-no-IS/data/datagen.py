#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

import os
import sys
import importlib.util

import hjson

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "../../main/data"))

from datagen_base import DataGeneratorBase, BANK_BYTES, BANKWIDTH  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]

# Load main-tiled-oscore's generator by file path (the local file is also named datagen.py, so a
# plain `from datagen import ...` would re-import THIS module -> circular import).
_spec = importlib.util.spec_from_file_location(
    "_oscore_datagen", os.path.abspath(os.path.join(_HERE, "../../main-tiled-oscore/data/datagen.py"))
)
_oscore_datagen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oscore_datagen)


class DataGenerator(_oscore_datagen.DataGenerator):
    APP_NAME = "P2-async-OS-no-IS"

    def __init__(self, **kwargs):
        DataGeneratorBase.__init__(self, self.APP_NAME, **kwargs)
        self.phase1_scalars = {}
        self.phase2_scalars = {}
        local = self.params_in_path(__file__)
        for key, value in hjson.loads(local.read_text()).items():
            self.kwargs.setdefault(key, value)

    def run(self):
        # Both phases are emitted (the shared helper.c references M1_* and M2_*); this app only
        # consumes the M2_* P2 constants, the async oscore_in ring scalars, and M33.
        self.save_params()
        self.check_tiling_constraints()
        self.build_Phase1_data()
        self.build_Phase2_data()
        self._emit_p2_rings()
        self._run_memory_model()

    def _emit_p2_rings(self):
        FP8 = 8
        su = self.seqLenUnroll
        dt_win = su * self.dtRank * FP8 // 8  # bytes of dt per window (packed stride)
        bc_win = su * (2 * self.dState) * FP8 // 8  # bytes of BC per window (ring stride)
        comb_win = su * self.xProjDim * FP8 // 8  # combined dt_BC window stride in L3 (= ts[2])
        broadcast = self.dInner_tile // self.delaySU  # BC re-reads per dInner-tile
        win_per_l_tile = self.L_tile // su

        # R2 (dt) PACKED: same temporal bounds as the base R2, only the L-window stride shrinks
        # from comb_win (xProjDim-wide) to dt_win (dtRank-wide).
        r2_tb = [self.dtRank * FP8 // self.switchcore_width, su, self.seqLen // su, self.dInner_tile // self.convUnroll]
        r2_ts = [(self.switchcore_width // 8) * su, BANK_BYTES, dt_win, 0]
        self.lines_params.append(f"int32_t M2_R2_tb_packed[] = {{{', '.join(map(str, r2_tb))}}};")
        self.lines_params.append(f"int32_t M2_R2_ts_packed[] = {{{', '.join(map(str, r2_ts))}}};")

        # R7 (BC) RING: inner dim walks nb_slots abutting slots (stride bc_win); the (stride-0) wrap
        # dim absorbs both the L-tile cycle and the broadcast re-reads.
        r7_tb = [
            (2 * self.dState * FP8) // (2 * self.suc_serial_width_BC),
            su,
            self.nb_slots * win_per_l_tile,
            (self.nb_l_tiles * broadcast) // self.nb_slots,
        ]
        r7_ts = [(2 * self.suc_serial_width_BC // 8) * su, BANK_BYTES, bc_win, 0]
        self.lines_params.append(f"int32_t M2_R7_tb_ring[] = {{{', '.join(map(str, r7_tb))}}};")
        self.lines_params.append(f"int32_t M2_R7_ts_ring[] = {{{', '.join(map(str, r7_ts))}}};")

        # ---- ConvFormat operands (x via R9, y via W2, z-read via R10): cg-gather rings. -----------
        # The SUC reads/writes these column-group-outer / L-inner: each delaySU-wide channel group
        # re-touches every L-tile (a stride-(convUnroll*su) slice of each window). The C refill GATHERS
        # one cg-slice per window into a contiguous block (2D DMA), so the streamer reads each slot
        # flat and the wrap absorbs the cg cycle, exactly like BC. cg_count == BC broadcast, so x/y/z
        # share BC's R11 cadence. Bounds = {sub, slots*win_per_l_tile (stride conv_slice), wrap, 1}.
        cg_count = self.dInner_tile // self.delaySU  # = BC broadcast
        conv_slice = self.convUnroll * su * FP8 // 8  # one cg slice per window (64 B), = d2 stride
        conv_win_stride = su * self.dInnerUnroll * FP8 // 8  # ConvFormat L-window stride in L3 (384 B)
        conv_sub = (self.convUnroll * su) // (BANKWIDTH // FP8)  # within-slice sub count (R9 tb[0] = 8)
        conv_ring_tb = [conv_sub, self.nb_slots * win_per_l_tile, (self.nb_l_tiles * cg_count) // self.nb_slots, 1]
        conv_ring_ts = [BANK_BYTES, conv_slice, 0, 0]
        for sname in ("R9", "R10", "W2"):
            self.lines_params.append(f"int32_t M2_{sname}_tb_ring[] = {{{', '.join(map(str, conv_ring_tb))}}};")
            self.lines_params.append(f"int32_t M2_{sname}_ts_ring[] = {{{', '.join(map(str, conv_ring_ts))}}};")

        # Scalars for the C-side strided-DMA extraction + ring refill pacing (paced on R11 = SUC out).
        for name, val in [
            ("dtBC_window_src_stride", comb_win),  # stride between windows in the combined L3 buffer
            ("dt_pack_window_bytes", dt_win),  # dt bytes per window (packed dst stride)
            ("dt_windows_total", self.seqLen // su),  # windows in the full packed dt buffer
            ("length_dt_packed", (self.seqLen // su) * dt_win),
            ("BC_l3_offset", dt_win),  # BC starts after dt within each combined window
            ("BC_window_bytes", bc_win),  # BC bytes per window
            ("BC_windows_per_l_tile", win_per_l_tile),
            ("length_BC_l_tile", win_per_l_tile * bc_win),
            ("BC_broadcast", broadcast),
            ("BC_n_visits", self.nb_l_tiles * broadcast),
            ("BC_gauge_step", self.L_tile * self.delaySU),  # R11 (SUC out) elements per BC L-tile read
        ]:
            self.lines_params.append(f"uint32_t M2_{name} = {val};")
        for name, val in [
            # ConvFormat cg-gather (shared by x/y/z): block = one cg-slice over win_per_l_tile windows.
            ("conv_cg_count", cg_count),
            ("conv_d2_count", self.dInnerUnroll // self.convUnroll),
            ("conv_slice_bytes", conv_slice),  # bytes of one cg-slice per window (= d2 stride)
            ("conv_window_src_stride", conv_win_stride),  # ConvFormat L-window stride in L3
            ("conv_d3_stride", self.seqLen * self.dInnerUnroll * FP8 // 8),  # d3' block stride within a tile
            ("conv_windows_per_l_tile", win_per_l_tile),
            ("length_conv_l_block", win_per_l_tile * conv_slice),
            ("conv_n_visits", self.nb_l_tiles * cg_count),
            ("conv_gauge_step", self.L_tile * self.delaySU),
            ("length_x_tile_full", self.seqLen * self.dInner_tile * FP8 // 8),  # per-dInner-tile x/z/y size
        ]:
            self.lines_params.append(f"uint32_t M2_{name} = {val};")

    def _run_memory_model(self):
        # Override the parent's: it resolves the app dir from its own __file__, which would load
        # main-tiled-oscore's model + params. Use THIS app's layout (memory_model.py next to here).
        import importlib.util

        app_dir = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("memory_model_p2", os.path.join(app_dir, "memory_model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from memory_model_base import run_model_from_datagen  # type: ignore[import]

        self.lines_params.append(run_model_from_datagen(mod.build_report, app_dir))


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
