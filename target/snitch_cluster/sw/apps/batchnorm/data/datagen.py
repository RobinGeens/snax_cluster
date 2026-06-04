#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Datagen for the standalone folded-BatchNorm + ReLU step (the SegFormer ConvModule
# tail that follows the 1x1-conv / OS-core GEMM). Emits the SIMD per-channel-affine
# streamer config and the lane-duplicated scale/shift vectors. See the matching scala
# reference DataGeneratorBatchnorm and the per-channel multiply in rmsnorm (step 6).

import pathlib
import sys
import os

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
# Path in Occamy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import DataGeneratorBase, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "batchnorm"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)

    def run(self):
        self.build_data()

    def build_data(self):
        # Name files M10_* to match the scala generator (SIMD_MUL_BF16 = mode id 10).
        mode_id = 10
        assert f"M{mode_id}_SIMD_MUL_BF16" in self.kwargs, "verify mode_id for SIMD_MUL_BF16"

        L = self.kwargs["seqLen"]
        D = self.kwargs["channels"]
        simdLanes = self.kwargs["simdLanes_bf16"]
        assert simdLanes == self.kwargs["seqLenUnroll"], "memory layout mismatch"
        assert L % simdLanes == 0, f"seqLen ({L}) must be a multiple of simdLanes ({simdLanes})"

        # SIMD_ADD_BF16 with the doRelu post-op bit set. doRelu = 1<<2 (verified against
        # einfft-tiled's M3_SIMD_ADD_BF16_RELU); the named SIMD_ADD_BF16 lacks the relu bit.
        simd_add_bf16_relu = self.kwargs["M8_SIMD_ADD_BF16"] | (1 << 2)

        # Per-channel affine walk (mirrors rmsnorm step 6 "x * weight"):
        #   x   : (L, D) BF16 in IS-core out layout = [L/lanes][D][lanes]
        #   vec : per-channel scalar duplicated over the lanes, held stationary across L
        xw = (
            [L // simdLanes, D],
            [D * simdLanes * BF16 // 8, simdLanes * BF16 // 8],
        )
        vec = (
            [L // simdLanes, D],
            [0, simdLanes * BF16 // 8],  # stay on this channel's lane block for all L
        )
        streamers = {
            "R7_xw": xw,    # activations (operand A)
            "R13_vec": vec,  # scale / shift (operand B), per-channel duplicated
            "W3_xw": xw,    # output
        }

        specs = [
            ("x", L * D * BF16 // 8),
            ("scale", simdLanes * D * BF16 // 8),  # duplicated over lanes
            ("shift", simdLanes * D * BF16 // 8),  # duplicated over lanes
            ("out", L * D * BF16 // 8),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas, "SIMD_ADD_BF16_RELU": simd_add_bf16_relu}

        test_data = {name: "uint16_t" for name in ("x", "scale", "shift", "out")}
        tests = {"out": L * D}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
