#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# VMamba SS2D datagen. Inherits the main program's Phase 1 + Phase 2 streamer
# configs verbatim; overrides test data loading to use per-direction VMamba data.

import pathlib
import sys
import os
import math

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import BANKWIDTH, DataGeneratorBase, BANK_BYTES, FP8, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "main_datagen",
    os.path.join(os.path.dirname(__file__), "../../main/data/datagen.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
MainDataGenerator = _mod.DataGenerator

DATA_MODE_ID = 2


class DataGenerator(MainDataGenerator):
    APP_NAME = "vmamba"

    def run(self):
        self.save_params()
        # Emit Phase 1 streamers + scalars (no test data — those are per-direction)
        self._build_Phase1_streamers_only()
        # Emit Phase 2 streamers + scalars (no test data)
        self._build_Phase2_streamers_only()
        # Emit per-direction VMamba test data
        self._build_vmamba_test_data()
        self.emit_l1_usage_comment()

    def _build_Phase1_streamers_only(self):
        """Same as main's build_Phase1_data but emits only streamers + scalars, no test_data."""
        mode_id = 1
        assert f"M{mode_id}_PHASE1" in self.kwargs, "verify mode_id"
        assert self.switchcore_width == BANKWIDTH

        streamers = {
            "R0": (
                [self.dModel, self.seqLen // self.seqLenUnroll, self.dInner // self.dInnerUnroll],
                [self.seqLenUnroll * FP8 // 8, self.dModel * self.seqLenUnroll * FP8 // 8, 0],
            ),
            "R1": (
                [self.downsized_dModel, self.seqLen // self.seqLenUnroll, self.dInner // self.dInnerUnroll],
                [self.gemm_weight_width // 8, 0, self.downsized_dModel * self.gemm_weight_width // 8],
            ),
            "R3": ([self.dConv * self.dInner * FP8 // BANKWIDTH], [BANK_BYTES]),
            "R4": ([self.dInner * FP8 // BANKWIDTH], [BANK_BYTES]),
            "R12": (
                [self.downsized_xProjDim, self.seqLen // self.seqLenUnroll, self.dInner // self.dInnerUnroll],
                [self.gemm_weight_width // 8, 0, self.downsized_xProjDim * self.gemm_weight_width // 8],
            ),
            "R13": (
                [(self.seqLen // self.seqLenUnroll) * self.xProjDim, self.dInner // self.dInnerUnroll],
                [self.seqLenUnroll * BF16 // 8, 0],
            ),
            "W1": ([self.seqLen * self.dInner * FP8 // BANKWIDTH], [BANK_BYTES]),
            "W3": (
                [(self.seqLen // self.seqLenUnroll) * self.xProjDim, self.dInner // self.dInnerUnroll],
                [self.seqLenUnroll * BF16 // 8, 0],
            ),
        }

        specs = [
            ("oscore_in", self.seqLen * self.dModel * FP8 // 8),
            ("oscore_weight", self.dModel * self.dInner * FP8 // 8),
            ("conv_weight", self.dInner * self.dConv * FP8 // 8),
            ("conv_bias", self.dInner * FP8 // 8),
            ("conv_out", self.seqLen * self.dInner * FP8 // 8),
            ("iscore_weight", self.dInner * self.xProjDim * FP8 // 8),
            ("iscore_out", self.seqLen * self.xProjDim * BF16 // 8),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}
        self.phase1_scalars = scalars.copy()

        # Emit streamers + scalars only, no test_data reading
        self.build_mode(mode_id, streamers, scalars=scalars, test_data={}, tests={})

        # Read shared Phase 1 weights from Scala generator (all under M2_ prefix)
        mid = DATA_MODE_ID
        for name in ("oscore_weight", "conv_weight", "conv_bias"):
            self.read_and_format_vector(mid, "uint8_t", name)
        self.read_and_format_vector(mid, "uint16_t", "iscore_bias")

    def _build_Phase2_streamers_only(self):
        """Same as main's build_Phase2_data but emits only streamers + scalars."""
        mode_id = 2
        assert f"M{mode_id}_PHASE2" in self.kwargs, "verify mode_id"
        assert self.switchcore_width == BANKWIDTH

        suc_parallel_widthA = self.dState * FP8
        switchcore_parallel_width_W1 = self.convUnroll * self.dConv * FP8
        switchcore_parallel_width_W2 = self.convUnroll * (self.dtRankUnroll - self.dConv) * FP8
        oscore_parallel_width_d = self.seqLenUnroll * self.dInnerUnroll * FP8
        iscore_parallel_width_d = self.seqLenUnroll * self.dInnerUnroll * FP8

        bounds_conv_to_suc = [
            (self.convUnroll * self.seqLenUnroll) // (BANKWIDTH // FP8),
            self.seqLen // self.seqLenUnroll,
            self.dInnerUnroll // self.convUnroll,
            self.dInner // self.dInnerUnroll,
        ]
        strides_conv_to_suc = [
            BANK_BYTES,
            self.seqLenUnroll * self.dInnerUnroll * FP8 // 8,
            self.convUnroll * self.seqLenUnroll * FP8 // 8,
            self.seqLen * self.dInnerUnroll * FP8 // 8,
        ]

        streamers = {
            "R0": (
                [self.dModel, self.seqLen // self.seqLenUnroll, self.dInner // self.dInnerUnroll],
                [self.seqLenUnroll * FP8 // 8, self.dModel * self.seqLenUnroll * FP8 // 8, 0],
            ),
            "R1": (
                [self.downsized_dModel, self.seqLen // self.seqLenUnroll, self.dInner // self.dInnerUnroll],
                [self.gemm_weight_width // 8, 0, self.downsized_dModel * self.gemm_weight_width // 8],
            ),
            "R2": (
                [self.dtRank * FP8 // self.switchcore_width, self.seqLenUnroll,
                 self.seqLen // self.seqLenUnroll, self.dInner // self.convUnroll],
                [(self.switchcore_width // 8) * self.seqLenUnroll, BANK_BYTES,
                 self.seqLenUnroll * self.xProjDim * FP8 // 8, 0],
                self.seqLenUnroll * BANK_BYTES,
            ),
            "R3": (
                [(self.dtRank // self.dtRankUnroll) * (self.dInner // self.convUnroll)
                 * (switchcore_parallel_width_W1 // self.switchcore_width)],
                [self.switchcore_width // 8],
            ),
            "R4": ([(self.dInner * FP8) // self.switchcore_width], [self.switchcore_width // 8]),
            "R5": (
                [(self.dtRank // self.dtRankUnroll) * (self.dInner // self.convUnroll)
                 * (switchcore_parallel_width_W2 // self.switchcore_width)],
                [self.switchcore_width // 8],
            ),
            "R6": ([self.dInner * (suc_parallel_widthA // self.suc_serial_width_A)],
                   [self.suc_serial_width_A // 8]),
            "R7": (
                [(2 * self.dState * FP8) // (2 * self.suc_serial_width_BC), self.seqLenUnroll,
                 self.seqLen // self.seqLenUnroll, self.dInner // self.delaySU],
                [(2 * self.suc_serial_width_BC // 8) * self.seqLenUnroll, BANK_BYTES,
                 self.seqLenUnroll * self.xProjDim * FP8 // 8, 0],
                self.seqLenUnroll * BANK_BYTES,
            ),
            "R8": ([self.dInner * FP8 // BANKWIDTH], [BANK_BYTES]),
            "R9": (bounds_conv_to_suc, strides_conv_to_suc),
            "R10": (bounds_conv_to_suc, strides_conv_to_suc),
            "R11": (
                [(self.dInner // self.dInnerUnroll) * (self.seqLen // self.seqLenUnroll)
                 * (iscore_parallel_width_d // self.iscore_serial_width)],
                [self.iscore_serial_width // 8],
            ),
            "R12": (
                [self.downsized_dModel, self.seqLen // self.seqLenUnroll, self.dInner // self.dInnerUnroll],
                [self.gemm_weight_width // 8, 0, self.downsized_dModel * self.gemm_weight_width // 8],
            ),
            "R13": (
                [(self.seqLen // self.seqLenUnroll) * self.dModel, self.dInner // self.dInnerUnroll],
                [self.seqLenUnroll * BF16 // 8, 0],
            ),
            "W0": (
                [(oscore_parallel_width_d // self.oscore_serial_width)
                 * (self.seqLen // self.seqLenUnroll) * (self.dInner // self.dInnerUnroll)],
                [self.oscore_serial_width // 8],
            ),
            "W2": (bounds_conv_to_suc, strides_conv_to_suc),
            "W3": (
                [(self.seqLen // self.seqLenUnroll) * self.dModel, self.dInner // self.dInnerUnroll],
                [self.seqLenUnroll * BF16 // 8, 0],
            ),
        }

        tensor_size = self.seqLen * self.dInner * FP8 // 8
        specs = [
            ("oscore_in", self.seqLen * self.dModel * FP8 // 8),
            ("oscore_weight", self.dModel * self.dInner * FP8 // 8),
            ("z", tensor_size),
            ("dt_BC", self.seqLen * self.xProjDim * FP8 // 8),
            ("dt_weight_1", self.dInner * (self.dtRank // self.dtRankUnroll) * self.dConv * FP8 // 8),
            ("dt_weight_2", self.dInner * (self.dtRank // self.dtRankUnroll)
             * (self.dtRankUnroll - self.dConv) * FP8 // 8),
            ("dt_bias", self.dInner * FP8 // 8),
            ("x", tensor_size),
            ("A", self.dInner * self.dState * FP8 // 8),
            ("D", self.dInner * FP8 // 8),
            ("y", tensor_size),
            ("iscore_weight", self.dModel * self.dInner * FP8 // 8),
            ("iscore_out", self.seqLen * self.dModel * BF16 // 8),
        ]

        lengths, deltas = self._collect_lengths_and_deltas(specs)
        suc_start_cnt, iscore_start_cnt = self.get_safe_to_start_delay()
        scalars = {
            **lengths, **deltas,
            "R10_start_cnt": suc_start_cnt,
            "R11_start_cnt": iscore_start_cnt,
            "dt_to_BC_offset": self.seqLenUnroll * self.dtRank * FP8 // 8,
        }
        self.phase2_scalars = scalars.copy()

        # Emit streamers + scalars, no test_data
        self.build_mode(mode_id, streamers, scalars=scalars, test_data={}, tests={})

        # Read shared Phase 2 data from Scala generator (M2_ prefix)
        mid = DATA_MODE_ID
        for name in ("oscore_weight_P2", "dt_weight_1", "dt_weight_2", "dt_bias",
                      "suc_A", "suc_D", "iscore_weight_P2"):
            self.read_and_format_vector(mid, "uint8_t", name)
        self.read_and_format_vector(mid, "uint16_t", "iscore_bias_P2")

    def _build_vmamba_test_data(self):
        """Read VMamba-specific data. Only per-direction INPUTS are K-stacked;
        per-direction verification goldens are dropped to keep the binary small."""
        mid = DATA_MODE_ID

        # Single input in flattenA (dir 0) — cross-scan done at runtime
        self.read_and_format_vector(mid, "uint8_t", "oscore_in")
        # Per-direction weights + cross-merge data
        for name in ("iscore_weight_K", "y_invperm_K"):
            self.read_and_format_vector(mid, "uint8_t", name)

        # Final output goldens (small, only L*D or L*dModel each)
        for name in ("y_merged_flat",):
            self.read_and_format_vector(mid, "uint8_t", name)
        self.format_test_samples(mid, "y_merged_flat", self.seqLen * self.dInner, 25)

        dir_LD = self.seqLen * self.dInner * FP8 // 8
        self.format("uint32_t", "dir_size_iscore_weight", self.dInner * self.xProjDim * FP8 // 8)
        self.format("uint32_t", "dir_size_y_invperm", dir_LD)

        # SIMD ADD streamer config for cross-merge (FP8 element-wise add)
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
        assert f"M{simd_mode_id}_SIMD_ADD_FP8" in self.kwargs, f"verify mode_id {simd_mode_id}"
        self.build_mode(simd_mode_id, simd_streamers, scalars={}, test_data={}, tests={})

        # ---- RMSNorm SIMD streamer configs (adapted from rmsnorm datagen) ----
        L = self.seqLen
        D = self.dInner
        simdLanes_bf16 = self.kwargs["simdLanes_bf16"]
        assert simdLanes_bf16 == self.seqLenUnroll

        bounds_LD = ([L * D // simdLanes_bf16], [simdLanes_bf16 * BF16 // 8])
        bounds_L = ([L // simdLanes_bf16], [simdLanes_bf16 * BF16 // 8])

        # Widen FP8→BF16 (from einfft)
        n_fp8_cycles = L * D // simd_fp8_lanes
        n_bf16_cycles = L * D // simdLanes_bf16
        rmsnorm_streamers = {
            "R7_widen": ([n_fp8_cycles, 1, 1, 1], [simd_fp8_lanes, 0, 0, 0],
                         [BANK_BYTES, 2 * BANK_BYTES]),
            "W3_widen": ([n_bf16_cycles, 1, 1, 1], [simdLanes_bf16 * BF16 // 8, 0, 0, 0],
                         [BANK_BYTES]),
            # Full x matrix (BF16)
            "R7_x": bounds_LD,
            "W3_x": bounds_LD,
            # RMS vector
            "R7_rms": bounds_L,
            "W3_rms": bounds_L,
            # x * rms broadcast
            "R13_x_rms": (
                [D, L // simdLanes_bf16],
                [0, simdLanes_bf16 * BF16 // 8],
            ),
            # x * weight
            "R7_x_w": ([L // simdLanes_bf16, D],
                       [D * simdLanes_bf16 * BF16 // 8, simdLanes_bf16 * BF16 // 8]),
            "R13_x_w": ([L // simdLanes_bf16, D],
                        [0, simdLanes_bf16 * BF16 // 8]),
            "W3_x_w": ([L // simdLanes_bf16, D],
                       [D * simdLanes_bf16 * BF16 // 8, simdLanes_bf16 * BF16 // 8]),
            # Narrow BF16→FP8
            "W3_narrow": ([n_fp8_cycles, 1, 1, 1], [simd_fp8_lanes, 0, 0, 0],
                          [BANK_BYTES]),
        }

        # RMSNorm buffers AFTER Phase 1 + Phase 2 region
        p1_end = self.phase1_scalars["addr_iscore_out"] + self.phase1_scalars["length_iscore_out"]
        p2_end = p1_end + max(v for k, v in self.phase2_scalars.items()
                              if k.startswith("addr_")) + max(
                              v for k, v in self.phase2_scalars.items() if k.startswith("length_"))
        rms_base = (p2_end + 63) & ~63  # align to 64

        rms_specs = [
            ("rms_x_bf16", L * D * BF16 // 8),     # widened x in BF16
            ("rms_const", simdLanes_bf16 * BF16 // 8),  # d_inverse / ones constant
            ("rms_weight", simdLanes_bf16 * D * BF16 // 8),  # weight duplicated
            ("rms_vec", L * BF16 // 8),             # RMS vector
        ]
        rms_lengths, rms_deltas = self._collect_lengths_and_deltas(rms_specs, base_offset=rms_base)
        rms_scalars = {**rms_lengths, **rms_deltas}

        # Use mode 12 for RMSNorm base streamers (SIMD_INPROD_BF16)
        rms_mode_id = 12
        self.build_mode(rms_mode_id, rmsnorm_streamers, scalars=rms_scalars, test_data={}, tests={})

        # Read RMSNorm data
        self.read_and_format_vector(mid, "uint16_t", "norm_weight")
        self.read_and_format_vector(mid, "uint8_t", "y_norm_flat")
        self.format_test_samples(mid, "y_norm_flat", L * D, 25)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
