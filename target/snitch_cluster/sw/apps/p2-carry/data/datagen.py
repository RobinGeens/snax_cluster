#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# P2 without the isCore, L-tiled with on-chip SSM state carry
# and a fused out-projection isgemm. See docs/dataflow/13_suc_carry.md.

import os
import sys
import math
import importlib.util

import hjson

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "../../main/data"))

from datagen_base import BANK_BYTES, BANKWIDTH, NUM_LOOPS, DataGeneratorBase, FP8  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]

# Reuse suc-carry's DataGenerator (P2 tensors + per-tile SUC dt/BC bounds + state carry bounds); loaded
# by path to avoid the module-name clash with this file.
_spec = importlib.util.spec_from_file_location(
    "_suc_carry_datagen", os.path.abspath(os.path.join(_HERE, "../../suc-carry/data/datagen.py"))
)
assert _spec is not None and _spec.loader is not None
_suc_carry_datagen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_suc_carry_datagen)


class DataGenerator(_suc_carry_datagen.DataGenerator):
    APP_NAME = "p2-carry"

    def __init__(self, **kwargs):
        DataGeneratorBase.__init__(self, self.APP_NAME, **kwargs)
        self.phase1_scalars = {}
        self.phase2_scalars = {}
        local = self.params_in_path(__file__)
        for key, value in hjson.loads(local.read_text()).items():
            self.kwargs.setdefault(key, value)

    def run(self):
        super().run()  # P2 tensors + suc-carry SUC dt/BC/state bounds (+ unused compact x/z/y bounds)
        self._emit_p2_carry()

    def _emit_streamer(self, name, tb, ts):
        tb = list(tb) + [1] * (NUM_LOOPS - len(tb))
        ts = list(ts) + [0] * (NUM_LOOPS - len(ts))
        self.lines_params.append(f"int32_t M2_{name}_tb[] = {{{', '.join(map(str, tb))}}};")
        self.lines_params.append(f"int32_t M2_{name}_ts[] = {{{', '.join(map(str, ts))}}};")

    def _emit_p2_carry(self):
        assert self.nb_tiles == 1, "p2-carry runs full dInner per invocation (nb_tiles must be 1)"
        su = self.seqLenUnroll
        dI = self.dInner
        nb_l_tiles = self.kwargs["nb_l_tiles"]
        L_tile = self.seqLen // nb_l_tiles
        assert self.seqLen % nb_l_tiles == 0 and L_tile % su == 0
        N_kern = dI // self.dInnerUnroll  # K-steps (full dInner)

        emit = self.lines_params.append

        # ---- convFormat conv<->suc bounds, scoped to L_tile (x=R9 in, z=R10 in, y=W2 out) ----
        # Same shape as main-tiled P2's bounds_conv_to_suc, with seqLen -> L_tile in the L-window count
        # (tb[1]) and the colblock stride (ts[3], which is now the per-d3 block size of an L_tile buffer).
        conv_tb = [
            (self.convUnroll * su) // (BANKWIDTH // FP8),
            L_tile // su,
            self.dInnerUnroll // self.convUnroll,
            N_kern,
        ]
        conv_ts = [
            BANK_BYTES,
            su * self.dInnerUnroll * FP8 // 8,
            self.convUnroll * su * FP8 // 8,
            L_tile * self.dInnerUnroll * FP8 // 8,
        ]
        for name in ("R9_lt", "R10_lt", "W2_lt"):
            self._emit_streamer(name, conv_tb, conv_ts)

        # ---- osCore in (R0), weight (R1), out z (W0), scoped to L_tile ----
        self._emit_streamer(
            "R0_lt", [self.dModel, L_tile // su, N_kern], [su * FP8 // 8, self.dModel * su * FP8 // 8, 0]
        )
        self._emit_streamer(
            "R1_lt",
            [self.downsized_dModel, L_tile // su, N_kern],
            [self.gemm_weight_width // 8, 0, self.downsized_dModel * self.gemm_weight_width // 8],
        )
        oscore_parallel_width_d = su * self.dInnerUnroll * FP8
        self._emit_streamer(
            "W0_lt",
            [(oscore_parallel_width_d // self.oscore_serial_width) * (L_tile // su) * N_kern],
            [self.oscore_serial_width // 8],
        )

        # ---- out-proj isCore, K-tiled (out stays FULL & accumulates in place; y/weight are contiguous
        # K-tile slices since K is convFormat-outermost). Non-final K-tiles run NO_REQUANT. See isgemm-tiled. ----
        BF16 = 16
        nb_op_k = self.kwargs.get("nb_op_k_tiles", 4)
        assert dI % (self.dInnerUnroll * nb_op_k) == 0, "dInner must be a multiple of dInnerUnroll*nb_op_k_tiles"
        op_dInner_tile = dI // nb_op_k
        K_t = op_dInner_tile // self.dInnerUnroll
        a_in_width = su * self.dInnerUnroll * FP8
        self._emit_streamer(
            "R11_kt",
            [K_t * (self.seqLen // su) * (a_in_width // self.iscore_serial_width)],
            [self.iscore_serial_width // 8],
        )
        self._emit_streamer(
            "R12_kt",
            [self.downsized_dModel, self.seqLen // su, K_t],
            [self.gemm_weight_width // 8, 0, self.downsized_dModel * self.gemm_weight_width // 8],
        )
        self._emit_streamer("W3_kt", [(self.seqLen // su) * self.dModel, K_t], [su * BF16 // 8, 0])

        # ---- convFormat L-tile geometry (x gather, y/z spill, all share the per-d3-colblock stride) ----
        colblock_count = dI // self.dInnerUnroll
        tcdm_colblock_stride = L_tile * self.dInnerUnroll * FP8 // 8  # one d3 block of an L_tile buffer
        l3_colblock_stride = self.seqLen * self.dInnerUnroll * FP8 // 8  # one d3 block of the full L3 tensor
        for name, val in [
            ("length_conv_l_tile", L_tile * dI * FP8 // 8),
            ("conv_colblock_count", colblock_count),
            ("conv_tcdm_colblock_stride", tcdm_colblock_stride),
            ("conv_l3_colblock_stride", l3_colblock_stride),
            ("conv_l_tile_l3_offset", tcdm_colblock_stride),  # per-L-tile offset within a d3 block
            ("length_oscore_in_l_tile", L_tile * self.dModel * FP8 // 8),
            ("oscore_in_l_tile_offset", L_tile * self.dModel * FP8 // 8),
            # out-proj K-tile geometry (y/weight tiles are contiguous slices of the full L3 tensors)
            ("NB_OP_K_TILES", nb_op_k),
            ("op_dInner_tile", op_dInner_tile),
            ("length_op_a_tile", self.seqLen * op_dInner_tile * FP8 // 8),  # y K-tile bytes
            ("length_op_b_tile", op_dInner_tile * self.dModel * FP8 // 8),  # weight K-tile bytes
        ]:
            emit(f"uint32_t M2_{name} = {val};")

        # osCore->SUC z safe-to-start (R10), scaled to one L-tile launch (main-tiled's formula on L_tile).
        gemm_window_cnt = L_tile // su
        gemm_total_tiles = (L_tile // su) * N_kern
        gemm_cycles = gemm_total_tiles * self.dModel
        suc_delta = (gemm_cycles - L_tile * dI) / self.dModel
        r10_start = min(math.ceil(max(gemm_window_cnt, suc_delta) * 1.2), gemm_total_tiles)
        emit(f"uint32_t M2_R10_start_cnt_lt = {int(r10_start)};")
        # NB_L_TILES and L_tile are already emitted by suc-carry's _emit_suc_carry (via super().run()).


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
