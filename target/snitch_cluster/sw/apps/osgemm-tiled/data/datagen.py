#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

import importlib.util
import os
import pathlib
import sys

import hjson

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
# Path in Occamy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import DataGeneratorBase, FP8  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]

APP_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("memory_model_osgemm_tiled", os.path.join(APP_DIR, "memory_model.py"))
memory_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(memory_model)


class DataGenerator(DataGeneratorBase):
    APP_NAME = "osgemm-tiled"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        # Not all parameters are propagated to scala, so read them from the local params hjson file
        with self.params_in_path(__file__).open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.build_osgemm_data()
        self._run_memory_model()

    def _run_memory_model(self):
        from memory_model_base import run_model_from_datagen  # type: ignore[import]

        self.lines_params.append(run_model_from_datagen(memory_model.build_report, APP_DIR))

    def build_osgemm_data(self):
        mode_id = 3
        assert f"M{mode_id}_OSGEMM" in self.kwargs, f"verify mode_id {mode_id} for OSGEMM"
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        assert (Mu, Nu) == (memory_model.MU, memory_model.NU)
        seqLen, K, dInner = self.kwargs["dim0"], self.kwargs["dim1"], self.kwargs["dim2"]
        oscore_serial_width = self.kwargs["oscore_serial_width"]
        plan = memory_model.plan(self.kwargs)

        a_in_width = Mu * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = Nu * FP8
        d_array_width = Mu * Nu * FP8
        assert d_array_width % oscore_serial_width == 0
        downsized_K = K * b_array_width // b_in_width
        assert downsized_K * b_in_width == K * b_array_width, f"K ({K}) x {b_array_width} not a multiple of {b_in_width}"
        d_beats = d_array_width // oscore_serial_width  # writer beats per (Mu x Nu) output tile

        # Descriptors of the largest launch
        M_c, N_t = max(plan["chunks"]), max(plan["tiles"])
        streamers = {
            "R0": ([K, M_c, N_t], [a_in_width // 8, K * a_in_width // 8, 0]),  # A chunk
            "R1": ([downsized_K, M_c, N_t], [b_in_width // 8, 0, downsized_K * b_in_width // 8]),  # B tile
            "W0": ([d_beats * M_c * N_t], [oscore_serial_width // 8]),  # D tile
        }

        len_a = seqLen * K * FP8 // 8
        len_b = K * dInner * FP8 // 8
        len_d = seqLen * dInner * FP8 // 8
        lengths, deltas = self._collect_lengths_and_deltas([("a", len_a), ("b", len_b), ("d", len_d)])
        scalars = {
            **lengths,
            **deltas,
            "length_a_chunk": plan["len_a_chunk"],
            "length_b_tile": plan["len_b_tile"],
            "length_d_tile": plan["len_d_tile"],
            "nb_inner_bufs": plan["nb_inner_bufs"],
            "nb_d_bufs": plan["nb_d_bufs"],
            "downsized_K": downsized_K,
            "d_beats": d_beats,
        }
        test_data = {name: "uint8_t" for name in ("A", "B", "D")}
        tests = {"D": seqLen * dInner}
        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)

        dram = memory_model.dram_bytes(self.kwargs)
        self.lines_params.append(f"// L3 DMA per run: read {dram['read']} B, write {dram['write']} B")


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
