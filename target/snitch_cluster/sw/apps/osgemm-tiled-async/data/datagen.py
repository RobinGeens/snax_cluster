#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Datagen for osgemm-tiled-async: a minimal single-osCore GEMM whose A input (oscore_in) rings
# asynchronously through TCDM, refilled from L3 (input-side ring; isolates the oscore_in refill
# of main-tiled-oscore). Design: docs/dataflow/09_async_tiling.md (input-side ring).

import pathlib
import sys
import os
import hjson

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
# Path in Occamy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import DataGeneratorBase, FP8  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "osgemm-tiled-async"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        # Not all parameters are propagated to scala, so read them from the local params hjson file
        local_params_path = self.params_in_path(__file__)
        with local_params_path.open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.build_osgemm_data()
        self._run_memory_model()

    def _run_memory_model(self):
        import importlib.util

        app_dir = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("memory_model_osgemm_tiled_async",
                                                      os.path.join(app_dir, "memory_model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from memory_model_base import run_model_from_datagen  # type: ignore[import]

        self.lines_params.append(run_model_from_datagen(mod.build_report, app_dir))

    def build_osgemm_data(self):
        mode_id = 3
        assert f"M{mode_id}_OSGEMM" in self.kwargs, f"verify mode_id {mode_id} for OSGEMM"
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        seqLen = self.kwargs["dim0"]
        dModel = self.kwargs["dim1"]
        dInner = self.kwargs["dim2"]
        nb_tiles = self.kwargs["nb_tiles"]
        oscore_serial_width = self.kwargs["oscore_serial_width"]

        # B1: oscore_in (A) L-tiling along seqLen — async refill of an nb_slots-slot ring.
        nb_l_tiles = self.kwargs["nb_l_tiles"]
        nb_slots = self.kwargs["nb_slots"]

        a_in_width = Mu * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = Nu * FP8
        d_array_width = Mu * Nu * FP8
        assert d_array_width % oscore_serial_width == 0, "d_array_width Must be divisible by oscore_serial_width"

        b_downsize_factor = b_in_width / b_array_width  # >= 1

        # In VersaCore naming convention
        M = seqLen // Mu
        K = dModel
        N = dInner // Nu
        downsized_K = int(K / b_downsize_factor)
        assert downsized_K == K / b_downsize_factor, f"downsized_K {K / b_downsize_factor} must be an integer"
        assert downsized_K * b_in_width == K * b_array_width

        # ------ dInner tiling (kept = 1 for the isolated refill test) -------------------
        assert N % nb_tiles == 0, (
            f"N ({N}) must be divisible by nb_tiles ({nb_tiles}); equivalently, "
            f"dim2 ({dInner}) must be a multiple of dInnerUnroll * nb_tiles ({Nu} * {nb_tiles})"
        )
        N_tile = N // nb_tiles
        dInner_tile = dInner // nb_tiles

        # ------ A seqLen L-tiling (the async ring) --------------------------------------
        assert seqLen % nb_l_tiles == 0, f"seqLen ({seqLen}) must be divisible by nb_l_tiles ({nb_l_tiles})"
        L_tile = seqLen // nb_l_tiles
        assert L_tile % Mu == 0, f"L_tile ({L_tile}) must be a multiple of seqLenUnroll ({Mu})"
        assert nb_slots >= 2, f"nb_slots ({nb_slots}) must be >= 2"
        assert (
            nb_l_tiles >= nb_slots and nb_l_tiles % nb_slots == 0
        ), f"nb_l_tiles ({nb_l_tiles}) must be >= nb_slots ({nb_slots}) and a multiple of it"
        # Refill lead must hide the ~250 cc L3->TCDM refill latency (see main-tiled-oscore/status.md).
        lead_cc = nb_slots * (L_tile // Mu) * dModel
        assert lead_cc >= 384, (
            f"async refill lead nb_slots*(L_tile/Mu)*dModel = {lead_cc} cc < 384 cc -> tiles will tear; "
            f"raise nb_slots, dModel, or L_tile"
        )

        # ------ Streamer bounds + strides ----------------------------------------------
        # R0 reads A from the nb_slots-slot ring via the stride-0 outer-loop trick:
        #   dim0 (K)          : walk dModel within an Mu-row
        #   dim1 (slot-walk)  : walk the nb_slots adjacent slots' seqLen-rows (stride = one Mu-row)
        #   dim2 (wrap)       : stride 0 -> rewind to the slot-group base, nb_l/nb_slots times
        #                       (the DM core refills the slots between wraps)
        #   dim3 (N_tile)     : stride 0 -> re-read all of A per dInner-unroll group
        streamers = {
            "R0": (
                [K, (nb_slots * L_tile) // Mu, nb_l_tiles // nb_slots, N_tile],
                [a_in_width // 8, K * a_in_width // 8, 0, 0],
            ),
            "R1": (  # Input B (whole; nb_tiles==1)
                [downsized_K, M, N_tile],
                [
                    b_in_width // 8,
                    0,  # M is irrelevant
                    downsized_K * b_in_width // 8,
                ],
            ),
            "W0": (  # Output D (whole)
                [(d_array_width // oscore_serial_width) * M * N_tile],
                [oscore_serial_width // 8],
            ),
        }

        # ------ Buffer sizes -----------------------------------------------------------
        len_a = M * K * a_in_width // 8
        len_b = K * N * b_array_width // 8
        len_d = M * N * d_array_width // 8
        assert len_b % nb_tiles == 0
        assert len_d % nb_tiles == 0
        len_b_tile = len_b // nb_tiles
        len_d_tile = len_d // nb_tiles

        specs = [
            ("a", len_a),
            ("b", len_b),
            ("d", len_d),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        tile_scalars = {
            "length_b_tile": len_b_tile,
            "length_d_tile": len_d_tile,
            "dim2_tile": dInner_tile,
            # A ring (per-slot byte size, L3 stride between L-tiles, gauge ticks per L-tile).
            "length_a_l_tile": L_tile * dModel * FP8 // 8,
            "oscore_in_l_offset": L_tile * dModel * FP8 // 8,
            "oscore_in_l_tile_gauge_step": L_tile // Mu,
        }
        scalars = {**lengths, **deltas, **tile_scalars}

        test_data = {name: "uint8_t" for name in ("A", "B", "D")}
        tests = {"D": seqLen * dInner}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
