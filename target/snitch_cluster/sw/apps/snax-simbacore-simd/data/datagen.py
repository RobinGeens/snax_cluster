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

from datagen_base import DataGeneratorBase, BF16, FP8  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "simd"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)

    def run(self):
        self.build_SIMD()

    def build_SIMD(self):
        mode_id = 8
        assert f"M{mode_id}_SIMD_ADD_BF16" in self.kwargs, "verify mode_id"

        dataLength = self.kwargs["numElem"]
        n_acc = self.kwargs["n_acc"]
        simdLanes_bf16 = self.kwargs["simdLanes_bf16"]
        simdLanes_fp8 = self.kwargs["simdLanes_fp8"]
        width = 2 * self.kwargs["suc_serial_width_BC"]
        dataLength_reduce = dataLength // n_acc

        # This must be equal to the other ports as well. Can we assert this?
        assert width == BF16 * simdLanes_bf16
        assert simdLanes_bf16 == self.kwargs["seqLenUnroll"], "memory layout mismatch"

        # We defined L/2 complex values in the scala generator
        bounds_and_strides_bf16 = ([dataLength // simdLanes_bf16], [width // 8])
        bounds_and_strides_reduce_bf16 = ([dataLength_reduce // simdLanes_bf16], [width // 8])
        bounds_and_strides_fp8 = ([dataLength // simdLanes_fp8], [width // 8])
        bounds_and_strides_reduce_fp8 = ([dataLength_reduce // simdLanes_fp8], [width // 8])

        streamers = {
            "R7_bf16": bounds_and_strides_bf16,  # Input A (real and imag are interleaved)
            "R13_bf16": bounds_and_strides_bf16,  # Input B
            "W3_bf16": bounds_and_strides_bf16,  # Output
            "W3_reduce_bf16": bounds_and_strides_reduce_bf16,  # Output with reduction dimension
            "W3_fp8": bounds_and_strides_fp8,  # Output with requantization
            "R7_fp8": bounds_and_strides_fp8,  # FP8 Input A
            "R13_fp8": bounds_and_strides_fp8,  # FP8 Input B
            "W3_reduce_fp8": bounds_and_strides_reduce_fp8,  # FP8 Output with reduction dimension
        }

        tensor_names = ("in_a", "in_b", "add_out", "sub_out", "mul_out", "cmul_out", "div_out", "sqrt_out")
        requant_tensor_names = ("mul_out",)
        reduce_tensor_names = ("inprod_out", "rms_out")  # Tensors with reduction dimension

        # Tests for BF16 SIMD core
        specs = [(f"{tensor_name}_bf16", dataLength * BF16 // 8) for tensor_name in tensor_names]
        specs += [(f"{tensor_name}_bf16", dataLength_reduce * BF16 // 8) for tensor_name in reduce_tensor_names]
        # Note the naming: bf16_requant means coming from bf16 core -> fp8 requant
        specs += [(f"{tensor_name}_bf16_requant", dataLength * FP8 // 8) for tensor_name in requant_tensor_names]

        # Tests for FP8 SIMD core
        specs += [(f"{tensor_name}_fp8", dataLength * FP8 // 8) for tensor_name in tensor_names]
        specs += [(f"{tensor_name}_fp8", dataLength_reduce * FP8 // 8) for tensor_name in reduce_tensor_names]
        specs += [(f"{tensor_name}_fp8_requant", dataLength * BF16 // 8) for tensor_name in requant_tensor_names]

        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}
        tests = {"out": dataLength, "out_reduce": dataLength_reduce}

        test_data = {f"{name}_bf16": "uint16_t" for name in tensor_names + reduce_tensor_names}
        test_data.update({f"{name}_fp8": "uint8_t" for name in tensor_names + reduce_tensor_names})
        test_data.update({f"{name}_bf16_requant": "uint8_t" for name in requant_tensor_names})
        test_data.update({f"{name}_fp8_requant": "uint16_t" for name in requant_tensor_names})

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
