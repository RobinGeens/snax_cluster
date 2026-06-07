#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Datagen for isgemm-tiled-async: an ISGeMM whose PSUM output rings asynchronously through TCDM
# to/from L3 (output-side analog of osgemm-tiled-async). Emits no-requant golden D for verification.
# Design: docs/dataflow/09_async_tiling.md ("Output-side variant").

import pathlib
import sys
import os
import hjson

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
# Path in Occamy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import DataGeneratorBase, FP8, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "isgemm-tiled-async"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        # nb_l_tiles / nb_slots are not propagated by the Scala generator; read from local params.
        local_params_path = self.params_in_path(__file__)
        with local_params_path.open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.build_isgemm_tiled_async_data()

    def build_isgemm_tiled_async_data(self):
        mode_id = 4
        assert f"M{mode_id}_ISGEMM" in self.kwargs, f"verify mode_id {mode_id} for ISGEMM"
        assert "M5_ISGEMM_NO_REQUANT" in self.kwargs, "verify mode 5 for ISGEMM_NO_REQUANT"
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        seqLen = self.kwargs["dim0"]
        dInner = self.kwargs["dim1"]
        dModel = self.kwargs["dim2"]
        iscore_serial_width = self.kwargs["iscore_serial_width"]
        nb_l_tiles = self.kwargs["nb_l_tiles"]
        nb_slots = self.kwargs["nb_slots"]

        a_in_width = Mu * Nu * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = Nu * FP8
        assert a_in_width % iscore_serial_width == 0, "a_in_width must be divisible by iscore_serial_width"

        b_downsize_factor = b_in_width / b_array_width  # >= 1
        downsized_dModel = int(dModel / b_downsize_factor)
        assert downsized_dModel == dModel / b_downsize_factor, f"{dModel / b_downsize_factor} must be an integer"
        assert downsized_dModel * b_in_width == dModel * b_array_width

        # ------ K (dInner) tiling: ONE dInnerUnroll step per invocation (K_t' = 1) ----------
        assert dInner % Nu == 0, f"dInner ({dInner}) must be a multiple of dInnerUnroll ({Nu})"
        nb_k_tiles = dInner // Nu  # SW-outer reduction loop; each invocation adds one K-step

        # ------ seqLen (M) L-tiling: the async psum ring -----------------------------------
        assert seqLen % nb_l_tiles == 0, f"seqLen ({seqLen}) must be divisible by nb_l_tiles ({nb_l_tiles})"
        L_tile = seqLen // nb_l_tiles
        assert L_tile % Mu == 0, f"L_tile ({L_tile}) must be a multiple of seqLenUnroll ({Mu})"
        MperL = L_tile // Mu  # M-tiles (16-row groups) per L-tile (slot)
        assert nb_slots >= 2, f"nb_slots ({nb_slots}) must be >= 2"
        assert nb_l_tiles >= nb_slots and nb_l_tiles % nb_slots == 0, (
            f"nb_l_tiles ({nb_l_tiles}) must be >= nb_slots ({nb_slots}) and a multiple of it"
        )

        M_total = seqLen // Mu  # total 16-row M-tiles
        psum_pos_per_l_tile = MperL * dModel  # W3/R13 output positions (out_d fires) per L-tile

        # ------ Streamer bounds + strides --------------------------------------------------
        # Single invocation: all L-tiles, ONE K-step (K_t' = 1).
        # R13/W3 walk the nb_slots-slot psum ring via the stride-0 outer-loop wrap (like R0 in
        # osgemm-tiled-async): inner = positions within one L-tile; mid = walk the nb_slots
        # adjacent slots; outer = stride-0 rewind, nb_l/nb_slots times (the DM core spills+reloads
        # the slots between wraps).
        psum_bounds_and_strides = (
            [
                psum_pos_per_l_tile,          # positions within one L-tile slot
                nb_slots,                     # walk the nb_slots adjacent slots
                nb_l_tiles // nb_slots,       # stride-0 wrap -> rewind to ring base
            ],
            [
                Mu * BF16 // 8,               # one 16-row position
                psum_pos_per_l_tile * Mu * BF16 // 8,  # one slot
                0,                            # rewind
            ],
        )

        streamers = {
            "R11": (  # iscore A (convFormat, K outermost) -> one K-step slice is contiguous over all M
                [M_total * (a_in_width // iscore_serial_width)],
                [iscore_serial_width // 8],
            ),
            "R12": (  # iscore weight for one K-step; reused across all M-tiles (stride 0)
                [downsized_dModel, M_total],
                [b_in_width // 8, 0],
            ),
            "R13": psum_bounds_and_strides,
            "W3": psum_bounds_and_strides,
        }

        # ------ Buffer sizes ----------------------------------------------------------------
        len_a = seqLen * dInner * FP8 // 8
        len_b = dInner * dModel * FP8 // 8
        len_cd = seqLen * dModel * BF16 // 8  # full psum (L3) / C bias / D_no_requant
        assert len_a % nb_k_tiles == 0, f"len_a ({len_a}) not divisible by nb_k_tiles ({nb_k_tiles})"
        assert len_b % nb_k_tiles == 0, f"len_b ({len_b}) not divisible by nb_k_tiles ({nb_k_tiles})"
        len_a_ktile = len_a // nb_k_tiles
        len_b_ktile = len_b // nb_k_tiles

        specs = [
            ("a", len_a),
            ("b", len_b),
            ("cd", len_cd),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        tile_scalars = {
            "nb_k_tiles": nb_k_tiles,
            "L_tile": L_tile,
            "length_a_ktile": len_a_ktile,
            "length_b_ktile": len_b_ktile,
            # psum ring: per-slot byte size == per-L-tile L3 stride; gauge ticks per L-tile.
            "length_psum_l_tile": psum_pos_per_l_tile * Mu * BF16 // 8,
            "iscore_out_l_tile_gauge_step": psum_pos_per_l_tile,
        }
        scalars = {**lengths, **deltas, **tile_scalars}

        test_data = {**{name: "uint8_t" for name in ("A", "B", "D")}, "C": "uint16_t", "D_no_requant": "uint16_t"}
        tests = {"D_no_requant": seqLen * dModel}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
