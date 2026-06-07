#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

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
    APP_NAME = "isgemm-tiled"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        # nb_tiles is not propagated by the Scala generator; read it from the local params hjson.
        local_params_path = self.params_in_path(__file__)
        with local_params_path.open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.build_isgemm_tiled_data()

    def build_isgemm_tiled_data(self):
        mode_id = 4
        assert f"M{mode_id}_ISGEMM" in self.kwargs, f"verify mode_id {mode_id} for ISGEMM"
        assert "M5_ISGEMM_NO_REQUANT" in self.kwargs, "verify mode 5 for ISGEMM_NO_REQUANT"
        seqLenUnroll = self.kwargs["seqLenUnroll"]
        dInnerUnroll = self.kwargs["dInnerUnroll"]
        seqLen = self.kwargs["dim0"]
        dInner = self.kwargs["dim1"]
        dModel = self.kwargs["dim2"]
        iscore_serial_width = self.kwargs["iscore_serial_width"]
        nb_tiles = self.kwargs["nb_tiles"]

        a_in_width = seqLenUnroll * dInnerUnroll * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = dInnerUnroll * FP8
        assert a_in_width % iscore_serial_width == 0, "a_in_width must be divisible by iscore_serial_width"

        # ------ Tiling along dInner / K (accumulating) ---------------------------------
        # Each tile invokes the iscore with K_t = dInner_tile / dInnerUnroll inner steps.
        # The CD buffer is shared across tiles (psum accumulates in place via R13 -> W3).
        # Non-final tiles run in M5_ISGEMM_NO_REQUANT to keep the psum in BF16; the final
        # tile switches to M4_ISGEMM to apply the FP8 requant.
        assert dInner % (dInnerUnroll * nb_tiles) == 0, (
            f"dInner ({dInner}) must be a multiple of dInnerUnroll * nb_tiles "
            f"({dInnerUnroll} * {nb_tiles} = {dInnerUnroll * nb_tiles})"
        )
        dInner_tile = dInner // nb_tiles
        K_t = dInner_tile // dInnerUnroll

        b_downsize_factor = b_in_width / b_array_width  # >= 1
        downsized_dModel = int(dModel / b_downsize_factor)
        assert downsized_dModel == dModel / b_downsize_factor, f"{dModel / b_downsize_factor} must be an integer"
        assert downsized_dModel * b_in_width == dModel * b_array_width

        # CD: FULL buffer, accumulates in place across tiles. Per invocation the streamer
        # sweeps the entire output K_t times (stride 0 in the K dim).
        psum_bounds_and_strides = (
            [
                (seqLen // seqLenUnroll) * dModel,  # full output sweep
                K_t,                                # K iterations within the per-tile invocation
            ],
            [
                seqLenUnroll * BF16 // 8,
                0,                                  # same buffer each K
            ],
        )

        streamers = {
            "R11": (  # iscore A. ConvFormat: K is outermost, so per-tile slice is contiguous.
                [K_t * (seqLen // seqLenUnroll) * (a_in_width // iscore_serial_width)],
                [iscore_serial_width // 8],
            ),
            "R12": (  # iscore weight. K-tiled; per-tile slice is contiguous.
                [
                    downsized_dModel,             # N
                    seqLen // seqLenUnroll,       # M
                    K_t,                          # K (per-tile)
                ],
                [
                    b_in_width // 8,
                    0,
                    downsized_dModel * b_in_width // 8,
                ],
            ),
            "R13": psum_bounds_and_strides,
            "W3": psum_bounds_and_strides,
        }

        # ------ Buffer sizes -----------------------------------------------------------
        len_a = seqLen * dInner * FP8 // 8
        len_b = dInner * dModel * FP8 // 8
        len_cd = seqLen * dModel * BF16 // 8  # c and d use same space
        assert len_a % nb_tiles == 0, f"len_a ({len_a}) not divisible by nb_tiles ({nb_tiles})"
        assert len_b % nb_tiles == 0, f"len_b ({len_b}) not divisible by nb_tiles ({nb_tiles})"
        len_a_tile = len_a // nb_tiles
        len_b_tile = len_b // nb_tiles

        specs = [
            ("a", len_a),
            ("b", len_b),
            ("cd", len_cd),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        tile_scalars = {
            "length_a_tile": len_a_tile,
            "length_b_tile": len_b_tile,
            "dInner_tile": dInner_tile,
        }
        scalars = {**lengths, **deltas, **tile_scalars}

        test_data = {**{name: "uint8_t" for name in ("A", "B", "D")}, "C": "uint16_t", "D_no_requant": "uint16_t"}
        tests = {"D": seqLen * dModel}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
