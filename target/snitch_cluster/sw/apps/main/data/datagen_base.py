#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>


import math
import re
import sys
import os
import inspect
import random
from abc import ABC

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
# Path in Occamy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))

from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from snax_utils import align_wide_addr  # noqa E402

import subprocess

try:
    chisel_ssm_path = subprocess.check_output(["bender", "path", "chisel-ssm"], text=True).strip()
except Exception as e:
    raise RuntimeError(f"Error getting chisel-ssm path through bender: {e}")

NUM_LOOPS = 4  # NOTE this must match the hjson config
BANKWIDTH = 64
BANK_BYTES = BANKWIDTH // 8
NB_TEST_SAMPLES = 25
BF16 = 16
FP8 = 8


class DataGeneratorBase(ABC):
    """Abstract base class to centralize data generation logic."""

    def __init__(self, app_name: str, **kwargs):
        self.kwargs = kwargs
        self.lines_params: list[str] = []
        self.lines_data: list[str] = []
        self.app_name = app_name
        self.gen_data_dir = os.path.join(chisel_ssm_path, "generated", "data", self.app_name)

    def run(self):
        pass

    def emit_header_file(self):
        """Generate all lines and return them as a string."""
        self.lines_params.append("#include <stdint.h>\n")
        self.format_params()
        self.format("uint32_t", "nb_test_samples", NB_TEST_SAMPLES)
        self.run()

        return "\n".join(self.lines_params + self.lines_data)

    def format(self, type: str, var_name: str, value: int):
        self.lines_params.append(format_scalar_definition(type, var_name, value))

    def format_int(self, value: int):
        """Format `value` as integer. The name will be the variable name in the caller's scope."""
        callers_local_vars = (
            inspect.currentframe().f_back.f_locals.items()  # pyright: ignore[reportOptionalMemberAccess]
        )
        variable_name = next((name for name, val in callers_local_vars if val is value), None)
        assert variable_name is not None, "Variable name not found"
        self.format("uint32_t", variable_name, value)

    def format_params(self):
        """Takes all parameters from the kwargs and formats them as integers."""
        for key, value in self.kwargs.items():
            self.format("uint32_t", key, value)

    def format_vector(self, type: str, var_name: str, value: list[int]):
        self.lines_data.append(format_vector_definition(type, var_name, value, alignment=8))

    def read_and_format_vector(self, mode_id: int, type: str, tensor_name: str):
        """Read data from GEN_DATA_DIR and format it as a vector. Filename is M<mode_id>_<tensor_name>.bin."""
        try:
            tensor_data = self._read_data_int(f"M{mode_id}_{tensor_name}.bin")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Error loading test data for {tensor_name}: {e}. Did you run the scala data generator and is the data directory correct?"
            )
        self.format_vector(type, f"M{mode_id}_{tensor_name}", tensor_data)

    def format_temporal_bounds_strides(self, streamer_name: str, mode_id: int, bounds: list[int], strides: list[int]):
        """Format temporal bounds and strides for a streamer by automatically naming the variables and adding defaults.
        bounds are from inner to outer loop.

        The naming is, e.g., M0_R3_tb or M2_W1_ts. tb = temporal bounds, ts = temporal strides.
        """
        assert all(
            s % BANK_BYTES == 0 for s in strides
        ), f"M{mode_id} {streamer_name}: Temporal strides {strides} not aligned to bank width"

        # Extend with defaults
        bounds = bounds + [1] * (NUM_LOOPS - len(bounds))
        strides = strides + [0] * (NUM_LOOPS - len(strides))
        # Save each bound/stride as a separate variable
        # for i in range(NUM_LOOPS):
        #     self.format("uint32_t", f"M{mode_id}_{streamer_name}_tb{i}", bounds[i])
        #     self.format("uint32_t", f"M{mode_id}_{streamer_name}_ts{i}", strides[i])
        # Group values into array
        self.lines_params += [f"int32_t M{mode_id}_{streamer_name}_tb[] = {{{', '.join(map(str, bounds))}}};"]
        self.lines_params += [f"int32_t M{mode_id}_{streamer_name}_ts[] = {{{', '.join(map(str, strides))}}};"]

    def format_spatial_strides(self, streamer_name: str, mode_id: int, strides: list[int]):
        assert isinstance(strides, list), f"{streamer_name}: Spatial strides must be a list, got {type(strides)}"
        if streamer_name.startswith("R7") and len(strides) != 2:
            raise ValueError(f"{streamer_name}: R7 must have 2 spatial strides, got {len(strides)}")
        elif not streamer_name.startswith("R7") and len(strides) != 1:
            raise ValueError(f"{streamer_name}: Must have 1 spatial stride, got {len(strides)}")

        # self.format("uint32_t", f"M{mode_id}_{streamer_name}_ss0", stride)
        self.lines_params += [f"int32_t M{mode_id}_{streamer_name}_ss[] = {{{', '.join(map(str, strides))}}};"]

    def _read_data_int(self, filename: str):
        """Read a vec from a file."""
        with open(os.path.join(self.gen_data_dir, filename), "r") as f:
            lines = f.readlines()
        data_lines = [line.strip() for line in lines if not line.startswith("#")]
        return [int(x) for x in data_lines]

    def format_test_samples(self, mode_id: int, tensor_name: str, tensor_size: int, nb_test_samples: int):
        """Format variables used to test only a subset of the output."""
        self.format_vector(
            "int32_t",
            f"M{mode_id}_test_samples_{tensor_name}",
            [random.randint(0, tensor_size - 1) for _ in range(nb_test_samples)],
        )

    def enable_channel(self, streamer_name: str, mode_id: int):
        self.format("uint32_t", f"M{mode_id}_{streamer_name}_en", 1)

    def disable_channel(self, streamer_name: str, mode_id: int):
        self.format("uint32_t", f"M{mode_id}_{streamer_name}_en", 0)

    @staticmethod
    def _collect_lengths_and_deltas(
        specs: list[tuple[str, int]], base_offset: int = 0
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Build length_/delta_ dictionaries from spec tuples. Each spec is a (name, length) tuple."""
        lengths = {}
        deltas = {}
        offset = base_offset
        for spec in specs:
            name, length = spec
            lengths[f"length_{name}"] = length
            deltas[f"addr_{name}"] = offset
            offset = align_wide_addr(offset + length)
        return lengths, deltas

    def __format_streamer(
        self,
        mode_id: int,
        streamers: dict[str, tuple[list[int], list[int]]] | dict[str, tuple[list[int], list[int], int | list[int]]],
        name: str,
    ):
        # Parse
        if len(streamers[name]) == 2:
            (bounds, strides) = streamers[name]  # pyright: ignore[reportAssignmentType]
            spatial_stride_raw = BANK_BYTES  # Default
        elif len(streamers[name]) == 3:
            (bounds, strides, spatial_stride_raw) = streamers[name]  # pyright: ignore[reportAssignmentType]

        # Validate
        self.validate_stride(name, strides[0])
        spatial_strides = self.process_spatial_stride(name, spatial_stride_raw)
        self.format_temporal_bounds_strides(name, mode_id, bounds, strides)
        self.format_spatial_strides(name, mode_id, spatial_strides)
        self.enable_channel(name, mode_id)

    def process_spatial_stride(self, name: str, spatial_stride_raw: list[int] | int):
        """User input can be integer or list. Output must be a list with the length equal to the spatial loop bounds.

        Only R7 has 2 loop bounds, all other streamers have 1 loop bound.
        """
        if not name.startswith("R7"):
            if isinstance(spatial_stride_raw, int):
                return [spatial_stride_raw]
            elif isinstance(spatial_stride_raw, list):
                assert (
                    len(spatial_stride_raw) == 1
                ), f"{name}: Spatial stride must be a list with length 1, got {len(spatial_stride_raw)}"
                return spatial_stride_raw

        else:
            if isinstance(spatial_stride_raw, int):
                # TODO we have hardcodeded loopsize 2
                return [spatial_stride_raw, 2 * spatial_stride_raw]
            elif isinstance(spatial_stride_raw, list):
                assert (
                    len(spatial_stride_raw) == 2
                ), f"{name}: Spatial stride must be a list with length 2, got {len(spatial_stride_raw)}"
                return spatial_stride_raw
        raise ValueError(f"{name}: Spatial stride must be an integer or a list, got {type(spatial_stride_raw)}")

    def build_mode(
        self,
        mode_id: int,
        streamers: dict[str, tuple[list[int], list[int]]] | dict[str, tuple[list[int], list[int], int]],
        scalars: dict[str, int],
        test_data: dict[str, str],
        tests: dict[str, int],
    ):
        """Process all settings of a single mode and convert them to C code.

        Args:
        - mode_id: digit to identify the mode
        - streamers: dict[streamer name, (temporal bounds, temporal strides)]
        - scalars: dict[scalar name, value]
        - test_data: dict[tensor name, dtype]
        - tests: dict[test name, tensor size]
        """
        # Iterate over all streamer names
        assert all(re.match(r"^(R([0-9]|1[0-3])|W([0-9]|1[0-3]))(_.*)?$", key) for key in streamers.keys())
        # TODO number of streamers is hardcoded here
        standard_names = [f"R{i}" for i in range(14)] + [f"W{i}" for i in range(4)]

        # Make sure all standard streamer names are formatted
        for name in standard_names:
            if name in streamers:
                self.__format_streamer(mode_id, streamers, name)
            else:
                self.disable_channel(name, mode_id)

        # Also format non-standard streamer names (e.g. "R0_alt")
        for name in streamers.keys():
            if name not in standard_names:
                self.__format_streamer(mode_id, streamers, name)

        # Format scalar values
        for key, value in scalars.items():
            self.format("uint32_t", f"M{mode_id}_{key}", int(value))

        for tensor, size in tests.items():
            self.format_test_samples(mode_id, tensor, tensor_size=size, nb_test_samples=NB_TEST_SAMPLES)

        # Read and format test data
        for tensor, dtype in test_data.items():
            self.read_and_format_vector(mode_id, dtype, tensor)

    def extend_unroll_factor_to_bankwidth(self, unroll_factor: int, elem_width: int):
        """Returns the smallest number greater than or equal unroll_factor, such that this number times elem_width is a multiple of bankwidth.

        Should be used when the input has a DecoupledDownsizer
        """
        return math.ceil(unroll_factor * elem_width / BANKWIDTH) * BANKWIDTH // elem_width

    def validate_stride(self, streamer_name: str, stride: int):
        """Validate that the stride respects the interconnect sparsity constraints"""
        # TODO: This should be read from the hjson config
        SPARSITY_CONFIG = {"R1": 4, "R12": 4, "R13": 4, "W4": 4}

        sparsity_factor = SPARSITY_CONFIG.get(streamer_name, 1)
        if stride % (sparsity_factor * BANK_BYTES) != 0:
            raise ValueError(f"{streamer_name}: Stride {stride} not aligned to {sparsity_factor}x bank width")
