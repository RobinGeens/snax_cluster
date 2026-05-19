#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# 3-way partitioned EinFFT (L = L1 * L2 * L3). See docs/dataflow/05_fft.md §5.3
# for the stage sequence and the no-software-reorder trick.

import pathlib
import sys
import os

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
# Path in Occamy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import DataGeneratorBase, FP8, BF16, BANK_BYTES  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "fft-3way"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)

    def run(self):
        self.build_data()

    def build_data(self):
        mode_id = 6
        assert f"M{mode_id}_ISGEMM_SQ" in self.kwargs, f"verify mode_id {mode_id} for ISGEMM_SQ"
        seqLenUnroll = self.kwargs["seqLenUnroll"]
        dInnerUnroll = self.kwargs["dInnerUnroll"]
        iscore_serial_width = self.kwargs["iscore_serial_width"]
        suc_serial_width_BC = self.kwargs["suc_serial_width_BC"]
        dModel = self.kwargs["dModel"]
        L = self.kwargs["seqLen"]
        L1 = self.kwargs["L1"]
        L2 = self.kwargs["L2"]
        L3 = self.kwargs["L3"]
        L1_padded = self.kwargs["L1_padded"]
        L2_padded = self.kwargs["L2_padded"]
        L3_padded = self.kwargs["L3_padded"]

        # GEMM dims for each partition
        M_1 = (2 * L1) // seqLenUnroll
        M_2 = (2 * L2) // seqLenUnroll
        M_3 = (2 * L3) // seqLenUnroll
        K_1 = L1_padded // dInnerUnroll
        K_2 = 2 * L2_padded // dInnerUnroll
        K_3 = 2 * L3_padded // dInnerUnroll
        N_1 = dModel * L2 * L3
        N_2 = dModel * L1 * L3
        N_3 = dModel * L1 * L2
        assert L1 * L2 * L3 == L
        assert L1 % seqLenUnroll == 0, f"L1 ({L1}) must be a multiple of {seqLenUnroll}"
        assert L1_padded // dInnerUnroll == L1 // seqLenUnroll
        assert (2 * suc_serial_width_BC) // 8 == 4 * BANK_BYTES, "SIMD input width must be 4 banks"

        # Downsizer accounting (matches fft 2-way)
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = seqLenUnroll * FP8
        b_downsize_factor = b_in_width / b_array_width  # >= 1
        downsized_N_1 = int(N_1 / b_downsize_factor)
        downsized_N_2 = int(N_2 / b_downsize_factor)
        downsized_N_3 = int(N_3 / b_downsize_factor)
        assert downsized_N_1 * b_in_width == N_1 * b_array_width
        assert downsized_N_2 * b_in_width == N_2 * b_array_width
        assert downsized_N_3 * b_in_width == N_3 * b_array_width

        cd_array_width = seqLenUnroll * BF16
        assert cd_array_width == b_in_width, "Memory layout mismatch"

        psum_bounds_and_strides_1 = ([M_1 * N_1, K_1], [cd_array_width // 8, 0])
        psum_bounds_and_strides_2 = ([M_2 * N_2, K_2], [cd_array_width // 8, 0])
        psum_bounds_and_strides_3 = ([M_3 * N_3, K_3], [cd_array_width // 8, 0])

        streamers = {
            # Step 1: partition 1.
            "R11_1": (
                [2 * L1 * L1_padded * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12_1": (
                [downsized_N_1, M_1, K_1],
                [b_in_width // 8, 0, downsized_N_1 * b_in_width // 8],
            ),
            "R13_1": psum_bounds_and_strides_1,
            "W3_1": psum_bounds_and_strides_1,
            # Step 2: hadamard 1 (CMul). Cycle order (inner→outer): m2, k1_low, k1_high, d.
            # The 4th level (k1_high) only fires when L1 > seqLenUnroll.
            "R7_2": (
                [L2, seqLenUnroll, L1 // seqLenUnroll, dModel],
                [
                    (L3 * FP8 // 8 // BANK_BYTES) * seqLenUnroll * BANK_BYTES,
                    BANK_BYTES,
                    seqLenUnroll * L2 * L3 * dModel * FP8 // 8,
                    L * FP8 // 8,
                ],
                [seqLenUnroll * BANK_BYTES, L * dModel * FP8 // 8],
            ),
            "R13_2": (
                [L2, seqLenUnroll, L1 // seqLenUnroll, dModel],
                [4 * BANK_BYTES, L2 * 4 * BANK_BYTES, seqLenUnroll * L2 * 4 * BANK_BYTES, 0],
            ),
            "W3_2": (
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 2B: SIMD NOOP deinterleave re/im → __flattenColMajor.
            "R7_2B": (
                [L * dModel * FP8 // (2 * suc_serial_width_BC), 2],
                [8 * BANK_BYTES, 2 * BANK_BYTES],
                [BANK_BYTES, 4 * BANK_BYTES],
            ),
            "W3_2B": (
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 3: partition 2.
            "R11_3": (
                [2 * L2 * 2 * L2_padded * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12_3": (
                [downsized_N_2, M_2, K_2],
                [b_in_width // 8, 0, downsized_N_2 * b_in_width // 8],
            ),
            "R13_3": psum_bounds_and_strides_2,
            "W3_3": psum_bounds_and_strides_2,
            # Step 4: hadamard 2 (CMul with twiddles2). Cycle order (inner→outer): k2, k1, d.
            # Twiddles2 is broadcast across k1 and d (middle/outer strides are 0).
            "R7_4": (
                [L2, L1, dModel, 1],
                [BANK_BYTES, (L3 // (BANK_BYTES // (FP8 // 8))) * L2 * BANK_BYTES, L * FP8 // 8, 0],
                [L2 * BANK_BYTES, L * dModel * FP8 // 8],
            ),
            "R13_4": (
                [L2, L1, dModel, 1],
                [4 * BANK_BYTES, 0, 0, 0],
            ),
            "W3_4": (
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 4B: SIMD NOOP deinterleave re/im → __flattenColMajor.
            "R7_4B": (
                [L * dModel * FP8 // (2 * suc_serial_width_BC), 2],
                [8 * BANK_BYTES, 2 * BANK_BYTES],
                [BANK_BYTES, 4 * BANK_BYTES],
            ),
            "W3_4B": (
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 5: partition 3.
            "R11_5": (
                [2 * L3 * 2 * L3_padded * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12_5": (
                [downsized_N_3, M_3, K_3],
                [b_in_width // 8, 0, downsized_N_3 * b_in_width // 8],
            ),
            "R13_5": psum_bounds_and_strides_3,
            "W3_5": psum_bounds_and_strides_3,
        }

        specs = [
            ("weight1", 2 * L1 * L1_padded * FP8 // 8),
            ("weight2", 2 * L2 * 2 * L2_padded * FP8 // 8),
            ("weight3", 2 * L3 * 2 * L3_padded * FP8 // 8),
            ("in", L * dModel * FP8 // 8),
            ("partition1_out", 2 * L * dModel * BF16 // 8),
            ("twiddles1", 2 * L * FP8 // 8),
            ("hadamard1_out", 2 * L * dModel * FP8 // 8),
            ("hadamard1_packed", 2 * L * dModel * FP8 // 8),
            ("partition2_out", 2 * L * dModel * BF16 // 8),
            ("twiddles2", 2 * L2 * L3 * FP8 // 8),
            ("hadamard2_out", 2 * L * dModel * FP8 // 8),
            ("hadamard2_packed", 2 * L * dModel * FP8 // 8),
            ("partition3_out", 2 * L * dModel * BF16 // 8),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}

        test_data = {
            **{
                name: "uint8_t"
                for name in (
                    "dft_weight1",
                    "dft_weight2",
                    "dft_weight3",
                    "dft_in",
                    "partition1_expected",
                    "twiddles1",
                    "hadamard1_expected",
                    "partition2_expected",
                    "twiddles2",
                    "hadamard2_expected",
                    "partition3_expected",
                )
            }
        }
        tests = {"expected": 2 * L * dModel}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
