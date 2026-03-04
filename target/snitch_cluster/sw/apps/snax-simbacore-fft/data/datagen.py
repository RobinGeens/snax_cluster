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
        dModel = self.kwargs["dModel"]
        L = self.kwargs["seqLen"]
        L1 = self.kwargs["L1"]
        L1_padded0 = self.kwargs["L1_padded0"]  # Weight kernel will be (L1_padded0 x L1_padded1)
        L1_padded1 = self.kwargs["L1_padded1"]  # Padded to (seqLenUnroll x dInnerUnroll)
        L2 = self.kwargs["L2"]

        M = (2 * L1) // seqLenUnroll
        K = L1_padded1 // dInnerUnroll
        N = dModel * L2
        assert 2 * L1 == L1_padded0 and L1 % seqLenUnroll == 0, f"2*L1 must be an unpadded multiple of {seqLenUnroll}"
        assert L1 * L2 == L
        assert L1_padded1 // dInnerUnroll == L1 // seqLenUnroll

        # For input B, we transfer more bits in less cycles due to downsizer
        # ! In ISGEMM_SQ, the array only takes seqLenUnroll < dInnerUnroll elements
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = seqLenUnroll * FP8
        b_downsize_factor = b_in_width / b_array_width  # >= 1
        downsized_N = int(N / b_downsize_factor)  # output channels
        assert downsized_N == N / b_downsize_factor, f"{dModel / b_downsize_factor} must be an integer"
        assert downsized_N * b_in_width == N * b_array_width

        cd_array_width = seqLenUnroll * BF16
        assert cd_array_width == b_in_width, "Memory layout mismatch"

        # First inject zeros, then (K-1) times the full output matrix
        # The initial values (C) can be at the same addresses as the output matrix
        psum_bounds_and_strides = (
            [
                M * N,  # one output matrix
                K,  # complete reduction dimension (K)
            ],
            [
                cd_array_width // 8,
                0,  # Go to same addresses again
            ],
        )

        streamers = {
            "R11": (  # iscore A: DFT weight, with padding. Stored in convFormat
                [L1_padded0 * L1_padded1 * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12": (  # iscore B: input sequence (coming from previous phase)
                [downsized_N, M, K],
                [  # This is unpadded, with tilesize seqLenUnroll. Hardware takes care of the padding.
                    b_in_width // 8,
                    0,
                    downsized_N * b_in_width // 8,
                ],
            ),
            "R13": psum_bounds_and_strides,
            "W3": psum_bounds_and_strides,
        }

        specs = [
            ("a", L1_padded0 * L1_padded1 * FP8 // 8),
            ("b", L * dModel * FP8 // 8),
            ("cd", 2 * L * dModel * BF16 // 8),  # c and d use same space
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}

        test_data = {**{name: "uint8_t" for name in ("dft_weight", "dft_in", "partition1_expected")}}
        tests = {"expected": 2 * L * dModel}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
