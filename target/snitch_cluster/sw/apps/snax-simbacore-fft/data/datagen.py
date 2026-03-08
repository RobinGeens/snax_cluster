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

from datagen_base import DataGeneratorBase, FP8, BF16, BANK_BYTES  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "fft"

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
        L1_padded = self.kwargs["L1_padded"]
        L2_padded = self.kwargs["L2_padded"]

        M_1 = (2 * L1) // seqLenUnroll
        M_2 = (2 * L2) // seqLenUnroll
        K_1 = L1_padded // dInnerUnroll
        K_2 = 2 * L2_padded // dInnerUnroll
        N_1 = dModel * L2
        N_2 = dModel * L1
        assert L1 % seqLenUnroll == 0, f"2*L1 must be an unpadded multiple of {seqLenUnroll}"
        assert L1 * L2 == L
        assert L1_padded // dInnerUnroll == L1 // seqLenUnroll
        assert (2 * suc_serial_width_BC) // 8 == 4 * BANK_BYTES, "SIMD input width must be 4 banks "

        # For input B, we transfer more bits in less cycles due to downsizer
        # ! In ISGEMM_SQ, the array only takes seqLenUnroll < dInnerUnroll elements
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = seqLenUnroll * FP8
        b_downsize_factor = b_in_width / b_array_width  # >= 1
        downsized_N_1 = int(N_1 / b_downsize_factor)  # output channels
        downsized_N_2 = int(N_2 / b_downsize_factor)  # output channels
        assert downsized_N_1 == N_1 / b_downsize_factor, f"{dModel / b_downsize_factor} must be an integer"
        assert downsized_N_1 * b_in_width == N_1 * b_array_width
        assert downsized_N_2 == N_2 / b_downsize_factor, f"{dModel / b_downsize_factor} must be an integer"
        assert downsized_N_2 * b_in_width == N_2 * b_array_width

        cd_array_width = seqLenUnroll * BF16
        assert cd_array_width == b_in_width, "Memory layout mismatch"

        # First inject zeros, then (K-1) times the full output matrix
        # The initial values (C) can be at the same addresses as the output matrix
        psum_bounds_and_strides_1 = (
            [
                M_1 * N_1,  # one output matrix
                K_1,  # complete reduction dimension (K)
            ],
            [
                cd_array_width // 8,
                0,  # Go to same addresses again
            ],
        )
        psum_bounds_and_strides_2 = (
            [
                M_2 * N_2,  # one output matrix
                K_2,  # complete reduction dimension (K)
            ],
            [
                cd_array_width // 8,
                0,  # Go to same addresses again
            ],
        )

        streamers = {
            #
            # Step 1: partition 1
            #
            "R11_1": (  # iscore A: DFT weight, with padding. Stored in convFormat
                [2 * L1 * L1_padded * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12_1": (  # iscore B: input sequence (coming from previous phase)
                [downsized_N_1, M_1, K_1],
                [  # This is unpadded, with tilesize seqLenUnroll. Hardware takes care of the padding.
                    b_in_width // 8,
                    0,
                    downsized_N_1 * b_in_width // 8,
                ],
            ),
            "R13_1": psum_bounds_and_strides_1,
            "W3_1": psum_bounds_and_strides_1,
            #
            # Step 2: Hadamard
            #
            "R7_2": (  # SIMD: input. must also un-stride the matrix
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],  # Real and imag
                [BANK_BYTES],  # We always take bank (i, i+L2, i+D, i+L2+D)
                [  # We now have to spatial stride loops
                    L1 * BANK_BYTES,  # Un-stride the matrix: take the next L1 elements
                    L * dModel * FP8 // 8,  # Take the imag part, one tensor further
                ],
            ),
            "R13_2": (  # Twiddles. Real and imag are interleaved
                [
                    2 * L * FP8 // (2 * suc_serial_width_BC),
                    dModel,
                ],
                [4 * BANK_BYTES, 0],
            ),
            "W3_2": (  # SIMD output: everything in-order. Re/im will be interleaved every 16 elements
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            #
            # Step 2B: retransform the CMUL output: packed real/imag (interleaved every 16 elements) -> separate real/imag
            #
            "R7_2B": (
                [
                    L * dModel * FP8 // (2 * suc_serial_width_BC),
                    2,  # First reals, then imags
                ],
                [
                    8 * BANK_BYTES,  # Skip 2 interleaved groups (ss[1] already reaches into next group)
                    2 * BANK_BYTES,  # First imag starts at bank 2
                ],
                [
                    BANK_BYTES,
                    4 * BANK_BYTES,  # Skips one interleaved group (16R + 16I = 32B)
                ],
            ),
            "W3_2B": (  # The idea is that we write everything in-order
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            #
            # Step 3: partition 2
            #
            "R11_3": (  # iscore A: DFT weight, with padding. Stored in convFormat
                [2 * L2 * 2 * L2_padded * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12_3": (  # iscore B: input sequence (coming from previous phase)
                [downsized_N_2, M_2, K_2],
                [  # This is unpadded, with tilesize seqLenUnroll. Hardware takes care of the padding.
                    b_in_width // 8,
                    0,
                    downsized_N_2 * b_in_width // 8,
                ],
            ),
            "R13_3": psum_bounds_and_strides_2,
            "W3_3": psum_bounds_and_strides_2,
        }

        specs = [
            ("weight1", 2 * L1 * L1_padded * FP8 // 8),
            ("weight2", 2 * L2 * 2 * L2_padded * FP8 // 8),
            ("in", L * dModel * FP8 // 8),
            ("partition1_out", 2 * L * dModel * BF16 // 8),  # c and d use same space
            ("twiddles", 2 * L * FP8 // 8),  # Real and imag
            ("hadamard_out", 2 * L * dModel * FP8 // 8),
            ("hadamard_reordered", 2 * L * dModel * FP8 // 8),
            ("partition2_out", 2 * L * dModel * BF16 // 8),  # c and d use same space
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}

        test_data = {
            **{
                name: "uint8_t"
                for name in (
                    "dft_weight1",
                    "dft_weight2",
                    "dft_in",
                    "partition1_expected",
                    "twiddles",
                    "hadamard_expected",
                    "hadamard_reordered",
                    "partition2_expected",
                )
            }
        }
        tests = {"expected": 2 * L * dModel}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
