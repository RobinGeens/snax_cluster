#!/usr/bin/env python3

# Copyright 2025 dInnerUnroll Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

import pathlib
import sys
import os

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../snax-simbacore-main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import DataGeneratorBase, FP8, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "rmsnorm"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)

    def run(self):
        self.build_data()

    def build_data(self):
        mode_id = 11
        assert f"M{mode_id}_SIMD_RMS" in self.kwargs, "verify mode_id"

        L = self.kwargs["seqLen"]
        D = self.kwargs["dModel"]
        simdLanes = self.kwargs["simdLanes"]
        assert simdLanes == self.kwargs["seqLenUnroll"], "memory layout mismatch"

        bounds_and_strides_LD = ([L * D // simdLanes], [simdLanes * BF16 // 8])
        bounds_and_strides_L = ([L // simdLanes], [simdLanes * BF16 // 8])
        streamers = {
            "R7_LD": bounds_and_strides_LD,
            "R7_L": bounds_and_strides_L,
            "W3_LD": bounds_and_strides_LD,
            "W3_L": bounds_and_strides_L,
        }

        specs = [
            ("x", L * D * BF16 // 8),
            ("d_inverse", simdLanes * BF16 // 8),
            ("weight", D * BF16 // 8),
            ("meanSq", L * BF16 // 8),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}

        test_data = {**{name: "uint16_t" for name in ("x", "weight", "meanSq")}}
        tests = {"expected": L}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
