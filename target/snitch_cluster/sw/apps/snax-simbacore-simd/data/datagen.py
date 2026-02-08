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

from datagen_base import DataGeneratorBase, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "simd"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)

    def run(self):
        self.build_SIMD()

    def build_SIMD(self):
        mode_id = 6  # also mode 7, 8, 9
        assert f"M{mode_id}_SIMD_ADD" in self.kwargs, "verify mode_id"

        dataLength = self.kwargs["numElem"]
        n_acc = self.kwargs["n_acc"]
        simdLanes = self.kwargs["simdLanes"]
        width = 2 * self.kwargs["suc_serial_width_BC"]
        dataLength_reduce = dataLength // n_acc

        # This must be equal to the other ports as well. Can we assert this?
        assert width == BF16 * simdLanes
        assert simdLanes == self.kwargs["seqLenUnroll"], "memory layout mismatch"

        # We defined L/2 complex values in the scala generator
        bounds_and_strides = ([dataLength // simdLanes], [width // 8])
        bounds_and_strides_reduce = ([dataLength_reduce // simdLanes], [width // 8])

        streamers = {
            "R7": bounds_and_strides,  # Input A (real and imag are interleaved)
            "R13": bounds_and_strides,  # Input B
            "W3": bounds_and_strides,  # Output
            "W3_reduce": bounds_and_strides_reduce,  # Output with reduction dimension
        }

        specs = [
            (tensor_name, dataLength * BF16 // 8)
            for tensor_name in ("in_a", "in_b", "add_out", "sub_out", "mul_out", "cmul_out", "div_out", "sqrt_out")
        ]
        specs += [(tensor_name, dataLength_reduce * BF16 // 8) for tensor_name in ("inprod_out", "rms_out")]

        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}
        tests = {"out": dataLength, "out_reduce": dataLength_reduce}
        test_data = {
            name: "uint16_t"
            for name in (
                "simd_a",
                "simd_b",
                "add_out",
                "sub_out",
                "mul_out",
                "cmul_out",
                "inprod_out",
                "rms_out",
                "div_out",
                "sqrt_out",
            )
        }

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
