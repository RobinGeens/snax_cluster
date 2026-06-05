#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Datagen for is-osgemm-tiled-async: osCore and isCore run concurrently (IS_OSGEMM), with the
# osCore A INPUT ringing through TCDM (refill, paced by R10_DELAY_GAUGE) and the isCore PSUM
# OUTPUT ringing through TCDM (spill+reload, paced by ISCORE_TILE_CNT). Combines the rings of
# osgemm-tiled-async and isgemm-tiled-async. Design: docs/dataflow/09_async_tiling.md.

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

from datagen_base import DataGeneratorBase, FP8, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "is-osgemm-tiled-async"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        # nb_l_tiles / nb_slots are not propagated by the Scala generators; read from local params.
        local_params_path = pathlib.Path(__file__).resolve().parent / "params_in.hjson"
        with local_params_path.open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.emit_combined_mode()
        self.build_osgemm_async_data()
        self.build_isgemm_async_data()

    def emit_combined_mode(self):
        """IS_OSGEMM mode value = OSGEMM | ISGEMM. We always run the NO_REQUANT variant: osCore still
        requantizes its FP8 output (the requant bit only gates isCore), while the isCore PSUM ring
        stays BF16 so it can spill/reload through L3."""
        is_osgemm = self.kwargs["M3_OSGEMM"] | self.kwargs["M4_ISGEMM"]
        requant_bit = self.kwargs["M4_ISGEMM"] ^ self.kwargs["M5_ISGEMM_NO_REQUANT"]
        iosgemm_no_requant = is_osgemm & ~requant_bit
        self.format("uint32_t", "IS_OSGEMM", is_osgemm)
        self.format("uint32_t", "IS_OSGEMM_NO_REQUANT", iosgemm_no_requant)

    def build_osgemm_async_data(self):
        """OSGEMM (mode 3): A_os(seqLen x dModel) x B_os(dModel x dInner) = D_os(seqLen x dInner).
        A INPUT rings asynchronously through TCDM; one dInnerUnroll N-slice per invocation (N_tile=1).
        Prefix: M3_."""
        mode_id = 3
        assert f"M{mode_id}_OSGEMM" in self.kwargs, f"verify mode_id {mode_id} for OSGEMM"
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        dInner = self.kwargs["dInner"]
        oscore_serial_width = self.kwargs["oscore_serial_width"]
        nb_l_tiles = self.kwargs["nb_l_tiles"]
        nb_slots = self.kwargs["nb_slots"]

        a_in_width = Mu * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = Nu * FP8
        d_array_width = Mu * Nu * FP8
        assert d_array_width % oscore_serial_width == 0, "d_array_width must be divisible by oscore_serial_width"

        b_downsize_factor = b_in_width / b_array_width  # >= 1

        M = seqLen // Mu
        K = dModel
        N = dInner // Nu
        downsized_K = int(K / b_downsize_factor)
        assert downsized_K == K / b_downsize_factor, f"downsized_K {K / b_downsize_factor} must be an integer"

        # One dInnerUnroll N-slice per invocation -> N_tile=1, nb_tiles == nb invocations == dInner/Nu.
        assert dInner % Nu == 0, f"dInner ({dInner}) must be a multiple of dInnerUnroll ({Nu})"
        nb_tiles = dInner // Nu
        N_tile = N // nb_tiles
        assert N_tile == 1
        dInner_tile = dInner // nb_tiles

        # A seqLen L-tiling (the async ring, shared with isCore).
        assert seqLen % nb_l_tiles == 0, f"seqLen ({seqLen}) must be divisible by nb_l_tiles ({nb_l_tiles})"
        L_tile = seqLen // nb_l_tiles
        assert L_tile % Mu == 0, f"L_tile ({L_tile}) must be a multiple of seqLenUnroll ({Mu})"
        assert nb_slots >= 2, f"nb_slots ({nb_slots}) must be >= 2"
        assert nb_l_tiles >= nb_slots and nb_l_tiles % nb_slots == 0, (
            f"nb_l_tiles ({nb_l_tiles}) must be >= nb_slots ({nb_slots}) and a multiple of it"
        )
        # Refill lead must hide the ~250 cc L3->TCDM refill latency (see main-tiled-oscore/status.md).
        lead_cc = nb_slots * (L_tile // Mu) * dModel
        assert lead_cc >= 384, (
            f"async refill lead nb_slots*(L_tile/Mu)*dModel = {lead_cc} cc < 384 cc -> tiles will tear; "
            f"raise nb_slots, dModel, or L_tile"
        )

        # R0 reads A from the nb_slots-slot ring via the stride-0 outer-loop wrap (see osgemm-tiled-async).
        streamers = {
            "R0": (
                [K, (nb_slots * L_tile) // Mu, nb_l_tiles // nb_slots, N_tile],
                [a_in_width // 8, K * a_in_width // 8, 0, 0],
            ),
            "R1": (  # Input B (one dInner tile, reloaded per invocation)
                [downsized_K, M, N_tile],
                [b_in_width // 8, 0, downsized_K * b_in_width // 8],
            ),
            "W0": (  # Output D (one dInner tile per invocation)
                [(d_array_width // oscore_serial_width) * M * N_tile],
                [oscore_serial_width // 8],
            ),
        }

        len_a = M * K * a_in_width // 8
        len_b = K * N * b_array_width // 8
        len_d = M * N * d_array_width // 8
        assert len_b % nb_tiles == 0
        assert len_d % nb_tiles == 0
        len_b_tile = len_b // nb_tiles
        len_d_tile = len_d // nb_tiles

        specs = [("a", len_a), ("b", len_b), ("d", len_d)]
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

    def build_isgemm_async_data(self):
        """ISGEMM (mode 4): A_is(seqLen x dInner) x B_is(dInner x dModel) + C = D_is(seqLen x dModel).
        PSUM OUTPUT rings asynchronously through TCDM; ONE dInnerUnroll K-step per invocation (K_t'=1).
        Emits no-requant golden D for verification. Prefix: M4_."""
        mode_id = 4
        assert f"M{mode_id}_ISGEMM" in self.kwargs, f"verify mode_id {mode_id} for ISGEMM"
        assert "M5_ISGEMM_NO_REQUANT" in self.kwargs, "verify mode 5 for ISGEMM_NO_REQUANT"
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        seqLen = self.kwargs["seqLen"]
        dInner = self.kwargs["dInner"]
        dModel = self.kwargs["dModel"]
        iscore_serial_width = self.kwargs["iscore_serial_width"]
        nb_l_tiles = self.kwargs["nb_l_tiles"]
        nb_slots = self.kwargs["nb_slots"]

        a_in_width = Mu * Nu * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = Nu * FP8
        assert a_in_width % iscore_serial_width == 0, "a_in_width must be divisible by iscore_serial_width"

        b_downsize_factor = b_in_width / b_array_width  # >= 1
        downsized_dModel = int(dModel / b_downsize_factor)
        assert downsized_dModel == dModel / b_downsize_factor, f"{dModel / b_downsize_factor} must be an integer"

        # K (dInner) tiling: ONE dInnerUnroll step per invocation (K_t' = 1).
        assert dInner % Nu == 0, f"dInner ({dInner}) must be a multiple of dInnerUnroll ({Nu})"
        nb_k_tiles = dInner // Nu  # SW-outer reduction loop; each invocation adds one K-step

        # seqLen (M) L-tiling: the async psum ring (shared with osCore).
        assert seqLen % nb_l_tiles == 0, f"seqLen ({seqLen}) must be divisible by nb_l_tiles ({nb_l_tiles})"
        L_tile = seqLen // nb_l_tiles
        assert L_tile % Mu == 0, f"L_tile ({L_tile}) must be a multiple of seqLenUnroll ({Mu})"
        MperL = L_tile // Mu
        assert nb_slots >= 2, f"nb_slots ({nb_slots}) must be >= 2"
        assert nb_l_tiles >= nb_slots and nb_l_tiles % nb_slots == 0, (
            f"nb_l_tiles ({nb_l_tiles}) must be >= nb_slots ({nb_slots}) and a multiple of it"
        )

        M_total = seqLen // Mu
        psum_pos_per_l_tile = MperL * dModel  # W3/R13 output positions per L-tile

        # R13/W3 walk the nb_slots-slot psum ring via the stride-0 outer-loop wrap (see isgemm-tiled-async).
        psum_bounds_and_strides = (
            [psum_pos_per_l_tile, nb_slots, nb_l_tiles // nb_slots],
            [Mu * BF16 // 8, psum_pos_per_l_tile * Mu * BF16 // 8, 0],
        )

        streamers = {
            "R11": (  # iscore A (convFormat, K outermost) -> one K-step slice contiguous over all M
                [M_total * (a_in_width // iscore_serial_width)],
                [iscore_serial_width // 8],
            ),
            "R12": (  # iscore weight for one K-step; reused across all M-tiles (stride 0)
                [downsized_dModel, M_total],
                [b_in_width // 8, 0],
            ),
            "R13": psum_bounds_and_strides,
            "W3": psum_bounds_and_strides,
        }

        len_a = seqLen * dInner * FP8 // 8
        len_b = dInner * dModel * FP8 // 8
        len_cd = seqLen * dModel * BF16 // 8  # full psum (L3) / C bias / D_no_requant
        assert len_a % nb_k_tiles == 0, f"len_a ({len_a}) not divisible by nb_k_tiles ({nb_k_tiles})"
        assert len_b % nb_k_tiles == 0, f"len_b ({len_b}) not divisible by nb_k_tiles ({nb_k_tiles})"
        len_a_ktile = len_a // nb_k_tiles
        len_b_ktile = len_b // nb_k_tiles

        specs = [("a", len_a), ("b", len_b), ("cd", len_cd)]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        tile_scalars = {
            "nb_k_tiles": nb_k_tiles,
            "L_tile": L_tile,
            "length_a_ktile": len_a_ktile,
            "length_b_ktile": len_b_ktile,
            # psum ring: per-slot byte size == per-L-tile L3 stride; gauge ticks per L-tile.
            "length_psum_l_tile": psum_pos_per_l_tile * Mu * BF16 // 8,
            "iscore_out_l_tile_gauge_step": psum_pos_per_l_tile,
        }
        scalars = {**lengths, **deltas, **tile_scalars}

        test_data = {**{name: "uint8_t" for name in ("A", "B", "D")}, "C": "uint16_t", "D_no_requant": "uint16_t"}
        tests = {"D_no_requant": seqLen * dModel}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
