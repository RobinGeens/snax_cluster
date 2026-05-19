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

from datagen_base import DataGeneratorBase, FP8  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "osgemm-tiled"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        # Not all parameters are propagated to scala, so read them from the local params hjson file
        local_params_path = pathlib.Path(__file__).resolve().parent / "params_in.hjson"
        with local_params_path.open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.build_osgemm_data()

    def build_osgemm_data(self):
        mode_id = 3
        assert f"M{mode_id}_OSGEMM" in self.kwargs, f"verify mode_id {mode_id} for OSGEMM"
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        seqLen = self.kwargs["dim0"]
        dModel = self.kwargs["dim1"]
        dInner = self.kwargs["dim2"]
        nb_tiles = self.kwargs["nb_tiles"]
        oscore_serial_width = self.kwargs["oscore_serial_width"]

        a_in_width = Mu * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = Nu * FP8
        d_array_width = Mu * Nu * FP8
        assert d_array_width % oscore_serial_width == 0, "d_array_width Must be divisible by oscore_serial_width"

        b_downsize_factor = b_in_width / b_array_width  # >= 1

        # In VersaCore naming convention
        M = seqLen // Mu
        K = dModel
        N = dInner // Nu
        # We transfer more bits in less cycles due to downsizer
        downsized_K = int(K / b_downsize_factor)
        assert downsized_K == K / b_downsize_factor, f"downsized_K {K / b_downsize_factor} must be an integer"
        assert downsized_K * b_in_width == K * b_array_width

        # ------ Tiling along dInner / N ------------------------------------------------
        # Each tile invokes the accelerator with N_tile inner steps. dInner_tile must
        # remain a multiple of Nu (= dInnerUnroll), i.e. N must be divisible by nb_tiles.
        assert N % nb_tiles == 0, (
            f"N ({N}) must be divisible by nb_tiles ({nb_tiles}); equivalently, "
            f"dim2 ({dInner}) must be a multiple of dInnerUnroll * nb_tiles ({Nu} * {nb_tiles})"
        )
        N_tile = N // nb_tiles
        dInner_tile = dInner // nb_tiles

        # ------ Streamer bounds + strides (per tile) -----------------------------------
        streamers = {
            # for n in N_tile (irrelevant dimension for A)
            #   for m in M
            #     for k in K
            #         parfor s in (tileSize / bankWidth)
            #             addr = k * tile_size + m * K * tile_size + s * bankWidth
            "R0": (  # Input A (shared across tiles, base ptr unchanged)
                [K, M, N_tile],
                [
                    a_in_width // 8,
                    K * a_in_width // 8,
                    0,
                ],
            ),
            "R1": (  # Input B (per tile, base ptr advances by length_b_tile each tile)
                [downsized_K, M, N_tile],
                [
                    b_in_width // 8,
                    0,  # M is irrelevant
                    downsized_K * b_in_width // 8,
                ],
            ),
            "W0": (  # Output D (per tile)
                [(d_array_width // oscore_serial_width) * M * N_tile],
                [oscore_serial_width // 8],
            ),
        }

        # ------ Buffer sizes -----------------------------------------------------------
        # Full sizes (for the L3 buffers M3_A / M3_B / M3_D)
        len_a = M * K * a_in_width // 8
        len_b = K * N * b_array_width // 8
        len_d = M * N * d_array_width // 8
        assert len_b % nb_tiles == 0
        assert len_d % nb_tiles == 0
        len_b_tile = len_b // nb_tiles
        len_d_tile = len_d // nb_tiles

        specs = [
            ("a", len_a),
            ("b", len_b),
            # ("c", M * N * d_array_width // 8),
            ("d", len_d),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        tile_scalars = {
            "length_b_tile": len_b_tile,
            "length_d_tile": len_d_tile,
            "dim2_tile": dInner_tile,
        }
        scalars = {**lengths, **deltas, **tile_scalars}

        test_data = {name: "uint8_t" for name in ("A", "B", "D")}
        tests = {"D": seqLen * dInner}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
