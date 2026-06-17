#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Tiled RMSNorm: identical math to the `rmsnorm` app, but the seqLen axis is split into
# nb_tiles L-tiles so arbitrarily large seqLen fits TCDM. RMSNorm is per-token (row-wise),
# so every L-tile is fully independent. The C side double-buffers the x tiles through L3
# (DMA-in / compute / DMA-out overlap). Streamer programs are sized for one L-tile.
# Design: docs/dataflow/14_rmsnorm_tiled.md

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

from datagen_base import DataGeneratorBase, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "rmsnorm-tiled"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        # nb_tiles is not propagated to scala, so read it from the local params hjson file.
        with open(self.params_in_path(__file__)) as f:
            for key, value in hjson.loads(f.read()).items():
                self.kwargs.setdefault(key, value)

    def run(self):
        self.build_data()
        self._run_memory_model()

    def _run_memory_model(self):
        import importlib.util
        app_dir = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("memory_model", os.path.join(app_dir, "memory_model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from memory_model_base import run_model_from_datagen
        comment = run_model_from_datagen(mod.build_report, app_dir)
        self.lines_params.append(comment)

    def build_data(self):
        mode_id = 12
        assert f"M{mode_id}_SIMD_INPROD_BF16" in self.kwargs, "verify mode_id"

        L = self.kwargs["seqLen"]
        D = self.kwargs["dModel"]
        nb_tiles = self.kwargs["nb_tiles"]
        simdLanes = self.kwargs["simdLanes_bf16"]
        assert simdLanes == self.kwargs["seqLenUnroll"], "memory layout mismatch"

        assert L % nb_tiles == 0, f"seqLen ({L}) must be divisible by nb_tiles ({nb_tiles})"
        Lt = L // nb_tiles
        assert Lt % simdLanes == 0, (
            f"L_tile ({Lt}) must be a multiple of simdLanes ({simdLanes}); "
            f"raise seqLen or lower nb_tiles"
        )

        # Per-tile streamer programs (one L-tile of Lt rows). Layout matches the `rmsnorm`
        # app with L -> Lt: x is [group][D][lane], rms is [group][lane], group = simdLanes rows.
        bounds_and_strides_LtD = ([Lt * D // simdLanes], [simdLanes * BF16 // 8])
        bounds_and_strides_Lt = ([Lt // simdLanes], [simdLanes * BF16 // 8])
        streamers = {
            # one x tile
            "R7_x": bounds_and_strides_LtD,
            "W3_x": bounds_and_strides_LtD,
            # processing the per-row RMS vector (one tile)
            "R7_rms": bounds_and_strides_Lt,
            "W3_rms": bounds_and_strides_Lt,
            # x * rms (rms broadcast over D, one value per token)
            "R13_x_rms": (
                [D, Lt // simdLanes],
                [0, simdLanes * BF16 // 8],
            ),
            # x * weight (weight per channel, stationary over L)
            "R7_x_w": (
                [Lt // simdLanes, D],
                [D * simdLanes * BF16 // 8, simdLanes * BF16 // 8],
            ),
            "R13_x_w": (
                [Lt // simdLanes, D],
                [0, simdLanes * BF16 // 8],
            ),
            "W3_x_w": (
                [Lt // simdLanes, D],
                [D * simdLanes * BF16 // 8, simdLanes * BF16 // 8],
            ),
        }

        # Per-tile byte lengths used by the C-side DMA + TCDM layout.
        scalars = {
            "length_x_tile": Lt * D * BF16 // 8,
            "length_rms_tile": Lt * BF16 // 8,
            "length_weight": simdLanes * D * BF16 // 8,  # weights duplicated over simd lanes
            "L_tile": Lt,
        }

        # Golden data (full L x D). x/weight DMA'd from L3; out is the golden output.
        test_data = {name: "uint16_t" for name in ("x", "weight", "out")}
        tests = {
            "expected": L * D,
        }

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
