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

        assert "M48_SIMD_FMA_HOLD_BF16_RELU" in self.kwargs, "FmaHold mode missing from generator output"

        specs = [
            ("x", L * D * BF16 // 8),
            ("scaleshift", 2 * simdLanes * D * BF16 // 8),  # per channel: scale block, shift block
            ("out", L * D * BF16 // 8),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)

        # Per-channel affine walk (bounds inner->outer; mirrors rmsnorm "x * weight"):
        #   x   : (L, D) BF16 in IS-core out layout = [L/lanes][D][lanes]
        #   vec : FmaHold b/c stream — one (scale, shift) lane-block pair per channel; the core
        #         holds the pair for the channel's n_acc = L/lanes outputs
        xw = (
            [L // simdLanes, D],
            [D * simdLanes * BF16 // 8, simdLanes * BF16 // 8],
        )
        vec = (
            [2, D],
            [simdLanes * BF16 // 8, 2 * simdLanes * BF16 // 8],
        )
        streamers = {
            "R7_xw": xw,    # activations (operand A)
            "R13_vec": vec,  # scale/shift interleaved (operands B/C)
            "W3_xw": xw,    # output
        }

        scalars = {**lengths, **deltas}

        # Interleave the generator's scale/shift vectors per channel: [D][2][lanes]
        scale = self._read_data_int(f"M{mode_id}_scale.bin")
        shift = self._read_data_int(f"M{mode_id}_shift.bin")
        scaleshift = []
        for ch in range(D):
            scaleshift += scale[ch * simdLanes:(ch + 1) * simdLanes]
            scaleshift += shift[ch * simdLanes:(ch + 1) * simdLanes]
        self.format_vector("uint16_t", f"M{mode_id}_scaleshift", scaleshift)

        test_data = {name: "uint16_t" for name in ("x", "out")}
        tests = {"out": L * D}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)
        self._run_memory_model()

    def _run_memory_model(self):
        import importlib.util

        app_dir = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("memory_model_batchnorm",
                                                      os.path.join(app_dir, "memory_model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from memory_model_base import run_model_from_datagen  # type: ignore[import]

        self.lines_params.append(run_model_from_datagen(mod.build_report, app_dir))


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
