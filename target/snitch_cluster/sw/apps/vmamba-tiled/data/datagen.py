#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# VMamba SS2D tiled datagen. Inherits main-tiled's tiling infrastructure
# (Phase 1 + Phase 2 streamer configs, tile sizes, K-step configs) and adds
# VMamba-specific data (per-direction inputs, cross-merge SIMD, RMSNorm SIMD).

import pathlib
import sys
import os
import math
import hjson

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import BANKWIDTH, DataGeneratorBase, BANK_BYTES, FP8, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "main_tiled_datagen",
    os.path.join(os.path.dirname(__file__), "../../main-tiled/data/datagen.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
MainTiledDataGenerator = _mod.DataGenerator

DATA_MODE_ID = 2


class DataGenerator(MainTiledDataGenerator):
    APP_NAME = "vmamba"

    def __init__(self, **kwargs):
        DataGeneratorBase.__init__(self, self.APP_NAME, **kwargs)
        self.phase1_scalars: dict[str, int] = {}
        self.phase2_scalars: dict[str, int] = {}
        local_params_path = pathlib.Path(__file__).resolve().parent / "params_in.hjson"
        with local_params_path.open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.save_params()
        self.check_tiling_constraints()
        self._build_Phase1_tiled_streamers()
        self._build_Phase2_tiled_streamers()
        self._build_vmamba_test_data()
        self.emit_l1_usage_comment()

    # =========================================================================
    # Phase 1 — tiled streamer configs (reuses main-tiled's _build_p1_streamers)
    # =========================================================================
    def _build_Phase1_tiled_streamers(self):
        mode_id = 1
        assert f"M{mode_id}_PHASE1" in self.kwargs, "verify mode_id"
        assert self.switchcore_width == BANKWIDTH

        K_i = self.dInner_tile // self.dInnerUnroll
        streamers_bulk = self._build_p1_streamers(K_i)

        K_step_deltas = {
            "R1_K_step_delta": self.downsized_dModel * self.gemm_weight_width // 8,
            "R3_K_step_delta": self.dConv * self.dInnerUnroll * FP8 // 8,
            "R4_K_step_delta": self.dInnerUnroll * FP8 // 8,
            "R12_K_step_delta": self.downsized_xProjDim * self.gemm_weight_width // 8,
            "W1_K_step_delta": self.seqLen * self.dInnerUnroll * FP8 // 8,
        }

        nb = self.nb_tiles
        len_oscore_in = self.seqLen * self.dModel * FP8 // 8
        len_oscore_weight = self.dModel * self.dInner * FP8 // 8
        len_conv_weight = self.dInner * self.dConv * FP8 // 8
        len_conv_bias = self.dInner * FP8 // 8
        len_conv_out = self.seqLen * self.dInner * FP8 // 8
        len_iscore_weight = self.dInner * self.xProjDim * FP8 // 8
        len_iscore_out = self.seqLen * self.xProjDim * BF16 // 8

        for v in (len_oscore_weight, len_conv_weight, len_conv_bias, len_conv_out, len_iscore_weight):
            assert v % nb == 0, f"Phase1 length {v} not divisible by nb_tiles {nb}"

        specs = [
            ("oscore_in", len_oscore_in),
            ("oscore_weight", len_oscore_weight),
            ("conv_weight", len_conv_weight),
            ("conv_bias", len_conv_bias),
            ("conv_out", len_conv_out),
            ("iscore_weight", len_iscore_weight),
            ("iscore_out", len_iscore_out),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)

        tile_scalars = {
            "dInner_tile": self.dInner_tile,
            "length_oscore_weight_tile": len_oscore_weight // nb,
            "length_conv_weight_tile": len_conv_weight // nb,
            "length_conv_bias_tile": len_conv_bias // nb,
            "length_conv_out_tile": len_conv_out // nb,
            "length_iscore_weight_tile": len_iscore_weight // nb,
        }
        scalars = {**lengths, **deltas, **tile_scalars, **K_step_deltas}
        self.phase1_scalars = scalars.copy()

        self.build_mode(mode_id, streamers_bulk, scalars=scalars, test_data={}, tests={})

        self._emit_alt_bounds(mode_id, self._build_p1_streamers(max(K_i - 1, 0)), "finalLead")
        self._emit_alt_bounds(mode_id, self._build_p1_streamers(1), "finalStep")

        mid = DATA_MODE_ID
        for name in ("oscore_weight", "conv_weight", "conv_bias"):
            self.read_and_format_vector(mid, "uint8_t", name)
        self.read_and_format_vector(mid, "uint16_t", "iscore_bias")

    # =========================================================================
    # Phase 2 — tiled streamer configs
    # =========================================================================
    def _build_Phase2_tiled_streamers(self):
        mode_id = 2
        assert f"M{mode_id}_PHASE2" in self.kwargs, "verify mode_id"
        assert self.switchcore_width == BANKWIDTH

        N_kern = self.dInner_tile // self.dInnerUnroll
        dInner_kern = self.dInner_tile

        suc_parallel_widthA = self.dState * FP8
        switchcore_parallel_width_W1 = self.convUnroll * self.dConv * FP8
        switchcore_parallel_width_W2 = self.convUnroll * (self.dtRankUnroll - self.dConv) * FP8
        oscore_parallel_width_d = self.seqLenUnroll * self.dInnerUnroll * FP8
        iscore_parallel_width_d = self.seqLenUnroll * self.dInnerUnroll * FP8

        bounds_conv_to_suc = [
            (self.convUnroll * self.seqLenUnroll) // (BANKWIDTH // FP8),
            self.seqLen // self.seqLenUnroll,
            self.dInnerUnroll // self.convUnroll,
            dInner_kern // self.dInnerUnroll,
        ]
        strides_conv_to_suc = [
            BANK_BYTES,
            self.seqLenUnroll * self.dInnerUnroll * FP8 // 8,
            self.convUnroll * self.seqLenUnroll * FP8 // 8,
            self.seqLen * self.dInnerUnroll * FP8 // 8,
        ]

        streamers = {
            "R0": (
                [self.dModel, self.seqLen // self.seqLenUnroll, N_kern],
                [self.seqLenUnroll * FP8 // 8, self.dModel * self.seqLenUnroll * FP8 // 8, 0],
            ),
            "R1": (
                [self.downsized_dModel, self.seqLen // self.seqLenUnroll, N_kern],
                [self.gemm_weight_width // 8, 0, self.downsized_dModel * self.gemm_weight_width // 8],
            ),
            "R2": (
                [self.dtRank * FP8 // self.switchcore_width, self.seqLenUnroll,
                 self.seqLen // self.seqLenUnroll, dInner_kern // self.convUnroll],
                [(self.switchcore_width // 8) * self.seqLenUnroll, BANK_BYTES,
                 self.seqLenUnroll * self.xProjDim * FP8 // 8, 0],
                self.seqLenUnroll * BANK_BYTES,
            ),
            "R3": (
                [(self.dtRank // self.dtRankUnroll) * (dInner_kern // self.convUnroll)
                 * (switchcore_parallel_width_W1 // self.switchcore_width)],
                [self.switchcore_width // 8],
            ),
            "R4": ([(dInner_kern * FP8) // self.switchcore_width], [self.switchcore_width // 8]),
            "R5": (
                [(self.dtRank // self.dtRankUnroll) * (dInner_kern // self.convUnroll)
                 * (switchcore_parallel_width_W2 // self.switchcore_width)],
                [self.switchcore_width // 8],
            ),
            "R6": ([dInner_kern * (suc_parallel_widthA // self.suc_serial_width_A)],
                   [self.suc_serial_width_A // 8]),
            "R7": (
                [(2 * self.dState * FP8) // (2 * self.suc_serial_width_BC), self.seqLenUnroll,
                 self.seqLen // self.seqLenUnroll, dInner_kern // self.delaySU],
                [(2 * self.suc_serial_width_BC // 8) * self.seqLenUnroll, BANK_BYTES,
                 self.seqLenUnroll * self.xProjDim * FP8 // 8, 0],
                self.seqLenUnroll * BANK_BYTES,
            ),
            "R8": ([dInner_kern * FP8 // BANKWIDTH], [BANK_BYTES]),
            "R9": (bounds_conv_to_suc, strides_conv_to_suc),
            "R10": (bounds_conv_to_suc, strides_conv_to_suc),
            "R11": (
                [(dInner_kern // self.dInnerUnroll) * (self.seqLen // self.seqLenUnroll)
                 * (iscore_parallel_width_d // self.iscore_serial_width)],
                [self.iscore_serial_width // 8],
            ),
            "R12": (
                [self.downsized_dModel, self.seqLen // self.seqLenUnroll, N_kern],
                [self.gemm_weight_width // 8, 0, self.downsized_dModel * self.gemm_weight_width // 8],
            ),
            "R13": (
                [(self.seqLen // self.seqLenUnroll) * self.dModel, N_kern],
                [self.seqLenUnroll * BF16 // 8, 0],
            ),
            "W0": (
                [(oscore_parallel_width_d // self.oscore_serial_width)
                 * (self.seqLen // self.seqLenUnroll) * (dInner_kern // self.dInnerUnroll)],
                [self.oscore_serial_width // 8],
            ),
            "W2": (bounds_conv_to_suc, strides_conv_to_suc),
            "W3": (
                [(self.seqLen // self.seqLenUnroll) * self.dModel, N_kern],
                [self.seqLenUnroll * BF16 // 8, 0],
            ),
        }

        tensor_size = self.seqLen * self.dInner * FP8 // 8
        nb = self.nb_tiles
        len_oscore_in = self.seqLen * self.dModel * FP8 // 8
        len_oscore_weight = self.dModel * self.dInner * FP8 // 8
        len_z = tensor_size
        len_dt_BC = self.seqLen * self.xProjDim * FP8 // 8
        len_dt_weight_1 = self.dInner * (self.dtRank // self.dtRankUnroll) * self.dConv * FP8 // 8
        len_dt_weight_2 = self.dInner * (self.dtRank // self.dtRankUnroll) * (self.dtRankUnroll - self.dConv) * FP8 // 8
        len_dt_bias = self.dInner * FP8 // 8
        len_x = tensor_size
        len_A = self.dInner * self.dState * FP8 // 8
        len_D = self.dInner * FP8 // 8
        len_y = tensor_size
        len_iscore_weight = self.dModel * self.dInner * FP8 // 8
        len_iscore_out = self.seqLen * self.dModel * BF16 // 8

        for v in (len_oscore_weight, len_z, len_dt_weight_1, len_dt_weight_2,
                  len_dt_bias, len_x, len_A, len_D, len_y, len_iscore_weight):
            assert v % nb == 0, f"Phase2 length {v} not divisible by nb_tiles {nb}"

        specs = [
            ("oscore_in", len_oscore_in),
            ("oscore_weight", len_oscore_weight),
            ("z", len_z),
            ("dt_BC", len_dt_BC),
            ("dt_weight_1", len_dt_weight_1),
            ("dt_weight_2", len_dt_weight_2),
            ("dt_bias", len_dt_bias),
            ("x", len_x),
            ("A", len_A),
            ("D", len_D),
            ("y", len_y),
            ("iscore_weight", len_iscore_weight),
            ("iscore_out", len_iscore_out),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        suc_start_cnt, iscore_start_cnt = self.get_safe_to_start_delay(self.dInner_tile)

        tile_scalars = {
            "dInner_tile": self.dInner_tile,
            "length_oscore_weight_tile": len_oscore_weight // nb,
            "length_z_tile": len_z // nb,
            "length_dt_weight_1_tile": len_dt_weight_1 // nb,
            "length_dt_weight_2_tile": len_dt_weight_2 // nb,
            "length_dt_bias_tile": len_dt_bias // nb,
            "length_x_tile": len_x // nb,
            "length_A_tile": len_A // nb,
            "length_D_tile": len_D // nb,
            "length_y_tile": len_y // nb,
            "length_iscore_weight_tile": len_iscore_weight // nb,
        }

        scalars = {
            **lengths, **deltas, **tile_scalars,
            "R10_start_cnt": suc_start_cnt,
            "R11_start_cnt": iscore_start_cnt,
            "dt_to_BC_offset": self.seqLenUnroll * self.dtRank * FP8 // 8,
        }
        self.phase2_scalars = scalars.copy()

        self.build_mode(mode_id, streamers, scalars=scalars, test_data={}, tests={})

        mid = DATA_MODE_ID
        for name in ("oscore_weight_P2", "dt_weight_1", "dt_weight_2", "dt_bias",
                      "suc_A", "suc_D", "iscore_weight_P2"):
            self.read_and_format_vector(mid, "uint8_t", name)
        self.read_and_format_vector(mid, "uint16_t", "iscore_bias_P2")

    # =========================================================================
    # VMamba-specific test data (per-direction inputs, cross-merge, RMSNorm)
    # =========================================================================
    def _build_vmamba_test_data(self):
        mid = DATA_MODE_ID

        self.read_and_format_vector(mid, "uint8_t", "oscore_in_K")
        for name in ("iscore_weight_K", "y_invperm_K"):
            self.read_and_format_vector(mid, "uint8_t", name)

        for name in ("y_merged_flat",):
            self.read_and_format_vector(mid, "uint8_t", name)
        self.format_test_samples(mid, "y_merged_flat", self.seqLen * self.dInner, 25)

        dir_LD = self.seqLen * self.dInner * FP8 // 8
        self.format("uint32_t", "dir_size_oscore_in", self.seqLen * self.dModel * FP8 // 8)
        self.format("uint32_t", "dir_size_iscore_weight", self.dInner * self.xProjDim * FP8 // 8)
        self.format("uint32_t", "dir_size_y_invperm", dir_LD)

        # ---- SIMD ADD streamers for cross-merge (FP8 element-wise add) ----
        simd_fp8_lanes = self.kwargs["simdLanes_fp8"]
        n_simd_cycles = self.seqLen * self.dInner // simd_fp8_lanes
        simd_streamers = {
            "R7": ([n_simd_cycles, 1, 1, 1], [simd_fp8_lanes, 0, 0, 0],
                   [BANK_BYTES, 2 * BANK_BYTES]),
            "R13": ([n_simd_cycles, 1, 1, 1], [simd_fp8_lanes, 0, 0, 0],
                    [BANK_BYTES]),
            "W3": ([n_simd_cycles, 1, 1, 1], [simd_fp8_lanes, 0, 0, 0],
                   [BANK_BYTES]),
        }
        simd_mode_id = 16
        assert f"M{simd_mode_id}_SIMD_ADD_FP8" in self.kwargs
        self.build_mode(simd_mode_id, simd_streamers, scalars={}, test_data={}, tests={})

        # ---- RMSNorm SIMD streamer configs ----
        L = self.seqLen
        D = self.dInner
        simdLanes_bf16 = self.kwargs["simdLanes_bf16"]
        assert simdLanes_bf16 == self.seqLenUnroll

        bounds_LD = ([L * D // simdLanes_bf16], [simdLanes_bf16 * BF16 // 8])
        bounds_L = ([L // simdLanes_bf16], [simdLanes_bf16 * BF16 // 8])

        n_fp8_cycles = L * D // simd_fp8_lanes
        n_bf16_cycles = L * D // simdLanes_bf16
        rmsnorm_streamers = {
            "R7_widen": ([n_fp8_cycles, 1, 1, 1], [simd_fp8_lanes, 0, 0, 0],
                         [BANK_BYTES, 2 * BANK_BYTES]),
            "W3_widen": ([n_bf16_cycles, 1, 1, 1], [simdLanes_bf16 * BF16 // 8, 0, 0, 0],
                         [BANK_BYTES]),
            "R7_x": bounds_LD,
            "W3_x": bounds_LD,
            "R7_rms": bounds_L,
            "W3_rms": bounds_L,
            "R13_x_rms": (
                [D, L // simdLanes_bf16],
                [0, simdLanes_bf16 * BF16 // 8],
            ),
            "R7_x_w": ([L // simdLanes_bf16, D],
                       [D * simdLanes_bf16 * BF16 // 8, simdLanes_bf16 * BF16 // 8]),
            "R13_x_w": ([L // simdLanes_bf16, D],
                        [0, simdLanes_bf16 * BF16 // 8]),
            "W3_x_w": ([L // simdLanes_bf16, D],
                       [D * simdLanes_bf16 * BF16 // 8, simdLanes_bf16 * BF16 // 8]),
            "W3_narrow": ([n_fp8_cycles, 1, 1, 1], [simd_fp8_lanes, 0, 0, 0],
                          [BANK_BYTES]),
        }

        # RMSNorm buffers placed after two merge buffers (which reuse TCDM after P1/P2)
        merge_buffer_end = 2 * L * D * FP8 // 8
        rms_base = (merge_buffer_end + 63) & ~63

        rms_specs = [
            ("rms_x_bf16", L * D * BF16 // 8),
            ("rms_const", simdLanes_bf16 * BF16 // 8),
            ("rms_weight", simdLanes_bf16 * D * BF16 // 8),
            ("rms_vec", L * BF16 // 8),
        ]
        rms_lengths, rms_deltas = self._collect_lengths_and_deltas(rms_specs, base_offset=rms_base)
        rms_scalars = {**rms_lengths, **rms_deltas}

        rms_mode_id = 12
        self.build_mode(rms_mode_id, rmsnorm_streamers, scalars=rms_scalars, test_data={}, tests={})

        self.read_and_format_vector(mid, "uint16_t", "norm_weight")
        self.read_and_format_vector(mid, "uint8_t", "y_norm_flat")
        self.format_test_samples(mid, "y_norm_flat", L * D, 25)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
