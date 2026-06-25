#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

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

# Superbank-pair size in bytes. TCDM is word-interleaved (bank = addr[7:3], 32 banks,
# 4 superbanks of 8). The low 128 B of every 256 B block map to banks 0-15 (superbanks
# 0-1), the high 128 B to banks 16-31 (superbanks 2-3).
# See docs/dataflow/20_double_gemm_conflict_free.md.
SKIP_BLOCK = 128


def skip128(bounds, strides):
    """Rewrite a streamer (temporal bounds + byte strides, dim0 = contiguous beat sweep)
    into the bank-partitioned skip-128 access pattern.

    The buffer is laid out so each 128 B logical block sits on a 256 B centre (the other
    half of every block belongs to the other GEMM). dim0 is split so it sweeps exactly one
    128 B block (= 16 banks of this GEMM's half); the block stride and every original outer
    stride are doubled (each must already be a multiple of 128). The result keeps every
    access of this streamer inside its 16-bank half, so it can never collide with the other
    core's streamers.
    """
    s0 = strides[0]
    assert SKIP_BLOCK % s0 == 0, f"beat stride {s0} must divide {SKIP_BLOCK}"
    inner = SKIP_BLOCK // s0  # beats per 128 B block
    assert bounds[0] % inner == 0, f"dim0 bound {bounds[0]} not divisible by {inner}"
    blocks = bounds[0] // inner
    new_bounds = [inner, blocks]
    new_strides = [s0, 2 * SKIP_BLOCK]  # block stride 128 -> physical 256
    for b, s in zip(bounds[1:], strides[1:]):
        assert s % SKIP_BLOCK == 0 or s == 0, f"outer stride {s} must be a multiple of {SKIP_BLOCK}"
        new_bounds.append(b)
        new_strides.append(2 * s)
    assert len(new_bounds) <= 4, f"skip-128 needs {len(new_bounds)} loop dims (max 4)"
    return new_bounds, new_strides


class DataGenerator(DataGeneratorBase):
    # Shares the OSGEMM/ISGEMM golden data with is-osgemm-tiled (identical arithmetic);
    # only the on-chip layout differs, so the sbt generator name stays is-osgemm-tiled.
    APP_NAME = "is-osgemm-tiled"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        local_params_path = self.params_in_path(__file__)
        with local_params_path.open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.emit_combined_mode()
        self.build_osgemm_data()
        self.build_isgemm_data()
        self._run_memory_model()

    def _run_memory_model(self):
        import importlib.util
        app_dir = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("memory_model", os.path.join(app_dir, "memory_model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from memory_model_base import run_model_from_datagen
        comment = run_model_from_datagen(mod.build_report, app_dir)
        self.lines_params.append(comment)

    def emit_combined_mode(self):
        """Compute IS_OSGEMM mode value = OSGEMM | ISGEMM (both cores, oscore output to mem)."""
        is_osgemm = self.kwargs["M3_OSGEMM"] | self.kwargs["M4_ISGEMM"]
        requant_bit = self.kwargs["M4_ISGEMM"] ^ self.kwargs["M5_ISGEMM_NO_REQUANT"]
        iosgemm_no_requant = is_osgemm & ~requant_bit
        self.format("uint32_t", "IS_OSGEMM", is_osgemm)
        self.format("uint32_t", "IS_OSGEMM_NO_REQUANT", iosgemm_no_requant)
        dInner_tile = self.kwargs["dInner"] // self.kwargs["nb_tiles"]
        self.format("uint32_t", "dInner_tile", dInner_tile)

    def build_osgemm_data(self):
        """OSGEMM (mode 3): A_os(seqLen x dModel) x B_os(dModel x dInner) = D_os(seqLen x dInner).
        Tiles along dInner (N dimension for oscore). Prefix: M3_."""
        mode_id = 3
        assert f"M{mode_id}_OSGEMM" in self.kwargs, f"verify mode_id {mode_id} for OSGEMM"
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        dInner = self.kwargs["dInner"]
        nb_tiles = self.kwargs["nb_tiles"]
        oscore_serial_width = self.kwargs["oscore_serial_width"]

        a_in_width = Mu * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = Nu * FP8
        d_array_width = Mu * Nu * FP8
        assert d_array_width % oscore_serial_width == 0

        b_downsize_factor = b_in_width / b_array_width
        M = seqLen // Mu
        K = dModel
        N = dInner // Nu
        downsized_K = int(K / b_downsize_factor)
        assert downsized_K == K / b_downsize_factor, f"downsized_K {K / b_downsize_factor} must be integer"
        assert N % nb_tiles == 0, f"N ({N}) must be divisible by nb_tiles ({nb_tiles})"

        N_tile = N // nb_tiles
        dInner_tile = dInner // nb_tiles

        streamers = {
            "R0": skip128(
                [K, M, N_tile],
                [a_in_width // 8, K * a_in_width // 8, 0],
            ),
            "R1": skip128(
                [downsized_K, M, N_tile],
                [b_in_width // 8, 0, downsized_K * b_in_width // 8],
            ),
            "W0": skip128(
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
        # skip-128 layout requires every DMA'd buffer to be a whole number of 128 B blocks.
        for nm, ln in (("a", len_a), ("b_tile", len_b_tile), ("d_tile", len_d_tile)):
            assert ln % SKIP_BLOCK == 0, f"OSGEMM {nm} length {ln} not a multiple of {SKIP_BLOCK}"

        specs = [("a", len_a), ("b", len_b), ("d", len_d)]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        tile_scalars = {"length_b_tile": len_b_tile, "length_d_tile": len_d_tile}
        scalars = {**lengths, **deltas, **tile_scalars}

        test_data = {name: "uint8_t" for name in ("A", "B", "D")}
        tests = {"D": seqLen * dInner}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)

    def build_isgemm_data(self):
        """ISGEMM (mode 4): A_is(seqLen x dInner) x B_is(dInner x dModel) + C = D_is(seqLen x dModel).
        Tiles along dInner (K dimension for iscore, accumulating). Prefix: M4_."""
        mode_id = 4
        assert f"M{mode_id}_ISGEMM" in self.kwargs, f"verify mode_id {mode_id} for ISGEMM"
        assert "M5_ISGEMM_NO_REQUANT" in self.kwargs
        seqLenUnroll = self.kwargs["seqLenUnroll"]
        dInnerUnroll = self.kwargs["dInnerUnroll"]
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        dInner = self.kwargs["dInner"]
        iscore_serial_width = self.kwargs["iscore_serial_width"]
        nb_tiles = self.kwargs["nb_tiles"]

        a_in_width = seqLenUnroll * dInnerUnroll * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = dInnerUnroll * FP8
        assert a_in_width % iscore_serial_width == 0

        assert dInner % (dInnerUnroll * nb_tiles) == 0, (
            f"dInner ({dInner}) must be a multiple of dInnerUnroll * nb_tiles "
            f"({dInnerUnroll} * {nb_tiles} = {dInnerUnroll * nb_tiles})"
        )
        dInner_tile = dInner // nb_tiles
        K_t = dInner_tile // dInnerUnroll

        b_downsize_factor = b_in_width / b_array_width
        downsized_dModel = int(dModel / b_downsize_factor)
        assert downsized_dModel == dModel / b_downsize_factor

        psum_bounds_and_strides = (
            [(seqLen // seqLenUnroll) * dModel, K_t],
            [seqLenUnroll * BF16 // 8, 0],
        )

        streamers = {
            "R11": skip128(
                [K_t * (seqLen // seqLenUnroll) * (a_in_width // iscore_serial_width)],
                [iscore_serial_width // 8],
            ),
            "R12": skip128(
                [downsized_dModel, seqLen // seqLenUnroll, K_t],
                [b_in_width // 8, 0, downsized_dModel * b_in_width // 8],
            ),
            "R13": skip128(*psum_bounds_and_strides),
            "W3": skip128(*psum_bounds_and_strides),
        }

        len_a = seqLen * dInner * FP8 // 8
        len_b = dInner * dModel * FP8 // 8
        len_cd = seqLen * dModel * BF16 // 8
        assert len_a % nb_tiles == 0
        assert len_b % nb_tiles == 0
        len_a_tile = len_a // nb_tiles
        len_b_tile = len_b // nb_tiles
        # skip-128 layout requires every DMA'd buffer to be a whole number of 128 B blocks.
        for nm, ln in (("a_tile", len_a_tile), ("b_tile", len_b_tile), ("cd", len_cd)):
            assert ln % SKIP_BLOCK == 0, f"ISGEMM {nm} length {ln} not a multiple of {SKIP_BLOCK}"

        specs = [("a", len_a), ("b", len_b), ("cd", len_cd)]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        tile_scalars = {"length_a_tile": len_a_tile, "length_b_tile": len_b_tile}
        scalars = {**lengths, **deltas, **tile_scalars}

        test_data = {**{name: "uint8_t" for name in ("A", "B", "D")}, "C": "uint16_t"}
        tests = {"D": seqLen * dModel}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
