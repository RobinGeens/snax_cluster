#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Element-wise FP8 add out = a + b, nb_tiles tiles of numElem elements that the C side double-buffers
# through TCDM. Golden data: the SIMD generator's FP8 add over one tile, which every tile reuses.
# Design: docs/dataflow/23_add_tiled.md

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
    APP_NAME = "add-tiled"

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
        mode_id = 8  # the SIMD generator names its files after M8 (SIMD_ADD_BF16)
        assert f"M{mode_id}_SIMD_ADD_BF16" in self.kwargs, "verify mode_id"

        N = self.kwargs["numElem"]
        nb_tiles = self.kwargs["nb_tiles"]
        lanes = self.kwargs["simdLanes_fp8"]
        beat = 2 * self.kwargs["suc_serial_width_BC"] // 8  # bytes per SIMD beat
        assert lanes * FP8 // 8 == beat, "one beat must hold all FP8 lanes"

        assert nb_tiles >= 1
        assert (N * FP8 // 8) % 64 == 0, f"tile ({N} elements) must be a multiple of 64 B"

        # One flat walk per port over one tile: R7 = a, R13 = b, W3 = out (in place over a)
        walk = ([N // lanes], [beat])
        streamers = {"R7_a": walk, "R13_b": walk, "W3_out": walk}

        scalars = {"length_tile": N * FP8 // 8}

        # One tile of a and b, DMA'd from L3 for every tile; add_out_fp8 is the golden output
        test_data = {name: "uint8_t" for name in ("in_a_fp8", "in_b_fp8", "add_out_fp8")}
        tests = {"out": N}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
