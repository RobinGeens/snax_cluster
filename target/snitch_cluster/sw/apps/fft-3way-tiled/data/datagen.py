#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Tiled 3-way partitioned EinFFT. See docs/dataflow/05_fft.md §5.4 for the
# per-phase tiling regime, TCDM overlay, and L3 spill rationale.

import pathlib
import sys
import os
import hjson

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
# Path in Occamy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import DataGeneratorBase, FP8, BF16, BANK_BYTES  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "fft-3way-tiled"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        local_params_path = pathlib.Path(__file__).resolve().parent / "params_in.hjson"
        with local_params_path.open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.build_data()
        self.emit_tcdm_usage_comment()

    def emit_tcdm_usage_comment(self):
        def align64(value: int) -> int:
            return (value + 63) & ~63

        L = self.kwargs["seqLen"]
        L1 = self.kwargs["L1"]
        L2 = self.kwargs["L2"]
        L3 = self.kwargs["L3"]
        L1_padded = self.kwargs["L1_padded"]
        L2_padded = self.kwargs["L2_padded"]
        L3_padded = self.kwargs["L3_padded"]
        dModel = self.kwargs["dModel"]
        nb_A = self.kwargs["nb_tiles_A"]
        nb_C = self.kwargs["nb_tiles_C"]

        len_weight1 = 2 * L1 * L1_padded * FP8 // 8
        len_weight2 = 2 * L2 * 2 * L2_padded * FP8 // 8
        len_weight3 = 2 * L3 * 2 * L3_padded * FP8 // 8
        len_twiddles1 = 2 * L * FP8 // 8
        len_twiddles2 = 2 * L2 * L3 * FP8 // 8
        len_in = L * dModel * FP8 // 8
        len_partition1_out = 2 * L * dModel * BF16 // 8
        len_hadamard1_out = 2 * L * dModel * FP8 // 8
        len_hadamard1_packed = 2 * L * dModel * FP8 // 8
        len_partition2_out = 2 * L * dModel * BF16 // 8
        len_hadamard2_out = 2 * L * dModel * FP8 // 8
        len_hadamard2_packed = 2 * L * dModel * FP8 // 8
        len_partition3_out = 2 * L * dModel * BF16 // 8

        always_live = (align64(len_weight1) + align64(len_weight2) + align64(len_weight3)
                       + align64(len_twiddles1) + align64(len_twiddles2))
        phase_a = (
            align64(len_in // nb_A)
            + align64(len_partition1_out // nb_A)
            + align64(len_hadamard1_out // nb_A)
            + align64(len_hadamard1_packed // nb_A)
        )
        # Phase B peak is during reorder2: hadamard2_out (overlays hadamard1_packed) +
        # partition2_out + hadamard2_packed.
        phase_b = (align64(len_hadamard2_out) + align64(len_partition2_out)
                   + align64(len_hadamard2_packed))
        phase_c = align64(len_hadamard2_packed // nb_C) + align64(len_partition3_out)
        peak = always_live + max(phase_a, phase_b, phase_c)

        untiled = (
            align64(len_weight1) + align64(len_weight2) + align64(len_weight3)
            + align64(len_twiddles1) + align64(len_twiddles2) + align64(len_in)
            + align64(len_partition1_out) + align64(len_hadamard1_out)
            + align64(len_hadamard1_packed) + align64(len_partition2_out)
            + align64(len_hadamard2_out) + align64(len_hadamard2_packed)
            + align64(len_partition3_out)
        )
        print(
            f"// TCDM peak usage (fft-3way-tiled): {peak} B ({peak/1024:.2f} KiB), "
            f"saves {untiled - peak} B ({(untiled-peak)/1024:.2f} KiB) vs un-tiled fft-3way "
            f"(was {untiled} B)"
        )
        print(f"// L3 buffers used: hadamard1_packed ({len_hadamard1_packed} B), "
              f"hadamard2_packed ({len_hadamard2_packed} B)")

    def build_data(self):
        mode_id = 6
        assert f"M{mode_id}_ISGEMM_SQ" in self.kwargs, f"verify mode_id {mode_id} for ISGEMM_SQ"
        seqLenUnroll = self.kwargs["seqLenUnroll"]
        dInnerUnroll = self.kwargs["dInnerUnroll"]
        iscore_serial_width = self.kwargs["iscore_serial_width"]
        suc_serial_width_BC = self.kwargs["suc_serial_width_BC"]
        dModel = self.kwargs["dModel"]
        L = self.kwargs["seqLen"]
        L1 = self.kwargs["L1"]
        L2 = self.kwargs["L2"]
        L3 = self.kwargs["L3"]
        L1_padded = self.kwargs["L1_padded"]
        L2_padded = self.kwargs["L2_padded"]
        L3_padded = self.kwargs["L3_padded"]
        nb_tiles_A = self.kwargs["nb_tiles_A"]
        nb_tiles_C = self.kwargs["nb_tiles_C"]

        # GEMM dims per partition
        M_1 = (2 * L1) // seqLenUnroll
        M_2 = (2 * L2) // seqLenUnroll
        M_3 = (2 * L3) // seqLenUnroll
        K_1 = L1_padded // dInnerUnroll
        K_2 = 2 * L2_padded // dInnerUnroll
        K_3 = 2 * L3_padded // dInnerUnroll
        N_1 = dModel * L2 * L3
        N_2 = dModel * L1 * L3
        N_3 = dModel * L1 * L2
        assert L1 * L2 * L3 == L
        assert L1 % seqLenUnroll == 0
        assert L1_padded // dInnerUnroll == L1 // seqLenUnroll
        assert (2 * suc_serial_width_BC) // 8 == 4 * BANK_BYTES

        # Only Phase A (dModel) and Phase C (K_3) are tiled; partition2 is un-tiled.
        assert dModel % nb_tiles_A == 0, f"dModel ({dModel}) must be divisible by nb_tiles_A ({nb_tiles_A})"
        assert K_3 % nb_tiles_C == 0, f"K_3 ({K_3}) must be divisible by nb_tiles_C ({nb_tiles_C})"
        dModel_tile = dModel // nb_tiles_A
        N_1_tile = dModel_tile * L2 * L3
        K_3_t = K_3 // nb_tiles_C
        dInner_3_tile = K_3_t * dInnerUnroll

        # Downsizer accounting
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = seqLenUnroll * FP8
        b_downsize_factor = b_in_width / b_array_width
        downsized_N_1 = int(N_1 / b_downsize_factor)
        downsized_N_2 = int(N_2 / b_downsize_factor)
        downsized_N_3 = int(N_3 / b_downsize_factor)
        downsized_N_1_tile = int(N_1_tile / b_downsize_factor)
        assert downsized_N_1 * b_in_width == N_1 * b_array_width
        assert downsized_N_2 * b_in_width == N_2 * b_array_width
        assert downsized_N_3 * b_in_width == N_3 * b_array_width
        assert downsized_N_1_tile == N_1_tile / b_downsize_factor

        cd_array_width = seqLenUnroll * BF16
        assert cd_array_width == b_in_width

        psum_p1_tile = ([M_1 * N_1_tile, K_1], [cd_array_width // 8, 0])
        psum_p2_full = ([M_2 * N_2, K_2], [cd_array_width // 8, 0])
        psum_p3_ktile = ([M_3 * N_3, K_3_t], [cd_array_width // 8, 0])

        streamers = {
            # Step 1: partition1 (Phase A, dModel-tiled).
            "R11_1": (
                [2 * L1 * L1_padded * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12_1": (
                [downsized_N_1_tile, M_1, K_1],
                [b_in_width // 8, 0, downsized_N_1_tile * b_in_width // 8],
            ),
            "R13_1": psum_p1_tile,
            "W3_1": psum_p1_tile,
            # Step 2: hadamard1 CMul. Cycle order (inner→outer): m2, k1_low, k1_high, d.
            # The k1_high level only fires when L1 > seqLenUnroll.
            "R7_2": (
                [L2, seqLenUnroll, L1 // seqLenUnroll, dModel_tile],
                [
                    (L3 * FP8 // 8 // BANK_BYTES) * seqLenUnroll * BANK_BYTES,
                    BANK_BYTES,
                    seqLenUnroll * L2 * L3 * dModel_tile * FP8 // 8,
                    L * FP8 // 8,
                ],
                [seqLenUnroll * BANK_BYTES, L * dModel_tile * FP8 // 8],
            ),
            "R13_2": (
                [L2, seqLenUnroll, L1 // seqLenUnroll, dModel_tile],
                [4 * BANK_BYTES, L2 * 4 * BANK_BYTES, seqLenUnroll * L2 * 4 * BANK_BYTES, 0],
            ),
            "W3_2": (
                [2 * L * dModel_tile * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 2B: SIMD NOOP deinterleave re/im (per-tile).
            "R7_2B": (
                [L * dModel_tile * FP8 // (2 * suc_serial_width_BC), 2],
                [8 * BANK_BYTES, 2 * BANK_BYTES],
                [BANK_BYTES, 4 * BANK_BYTES],
            ),
            "W3_2B": (
                [2 * L * dModel_tile * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 3: partition2 (Phase B, un-tiled).
            "R11_3": (
                [2 * L2 * 2 * L2_padded * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12_3": (
                [downsized_N_2, M_2, K_2],
                [b_in_width // 8, 0, downsized_N_2 * b_in_width // 8],
            ),
            "R13_3": psum_p2_full,
            "W3_3": psum_p2_full,
            # Step 4: hadamard2 CMul. Cycle order (inner→outer): k2, k1, d.
            "R7_4": (
                [L2, L1, dModel, 1],
                [BANK_BYTES, (L3 // (BANK_BYTES // (FP8 // 8))) * L2 * BANK_BYTES, L * FP8 // 8, 0],
                [L2 * BANK_BYTES, L * dModel * FP8 // 8],
            ),
            "R13_4": (
                [L2, L1, dModel, 1],
                [4 * BANK_BYTES, 0, 0, 0],
            ),
            "W3_4": (
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 4B: SIMD NOOP deinterleave re/im.
            "R7_4B": (
                [L * dModel * FP8 // (2 * suc_serial_width_BC), 2],
                [8 * BANK_BYTES, 2 * BANK_BYTES],
                [BANK_BYTES, 4 * BANK_BYTES],
            ),
            "W3_4B": (
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 5: partition3 (Phase C, K-axis tiled).
            "R11_5": (
                [(2 * L3 * 2 * L3_padded * FP8 // iscore_serial_width) // nb_tiles_C],
                [iscore_serial_width // 8],
            ),
            "R12_5": (
                [downsized_N_3, M_3, K_3_t],
                [b_in_width // 8, 0, downsized_N_3 * b_in_width // 8],
            ),
            "R13_5": psum_p3_ktile,
            "W3_5": psum_p3_ktile,
        }

        # ---------- Buffer sizes ------------------------------------------------
        len_weight1 = 2 * L1 * L1_padded * FP8 // 8
        len_weight2 = 2 * L2 * 2 * L2_padded * FP8 // 8
        len_weight3 = 2 * L3 * 2 * L3_padded * FP8 // 8
        len_in = L * dModel * FP8 // 8
        len_twiddles1 = 2 * L * FP8 // 8
        len_twiddles2 = 2 * L2 * L3 * FP8 // 8
        len_partition1_out = 2 * L * dModel * BF16 // 8
        len_hadamard1_out = 2 * L * dModel * FP8 // 8
        len_hadamard1_packed = 2 * L * dModel * FP8 // 8
        len_partition2_out = 2 * L * dModel * BF16 // 8
        len_hadamard2_out = 2 * L * dModel * FP8 // 8
        len_hadamard2_packed = 2 * L * dModel * FP8 // 8
        len_partition3_out = 2 * L * dModel * BF16 // 8

        # Phase A divisibility (dModel-axis).
        for name, value in (
            ("in", len_in),
            ("partition1_out", len_partition1_out),
            ("hadamard1_out", len_hadamard1_out),
            ("hadamard1_packed", len_hadamard1_packed),
        ):
            assert value % nb_tiles_A == 0, (
                f"length_{name} ({value}) not divisible by nb_tiles_A ({nb_tiles_A})")
        # Phase C divisibility (K-axis).
        for name, value in (
            ("hadamard2_packed", len_hadamard2_packed),
            ("weight3", len_weight3),
        ):
            assert value % nb_tiles_C == 0, (
                f"length_{name} ({value}) not divisible by nb_tiles_C ({nb_tiles_C})")

        # Phase A per-tile sizes (dModel-axis).
        len_in_tile = len_in // nb_tiles_A
        len_partition1_out_tile = len_partition1_out // nb_tiles_A
        len_hadamard1_out_tile = len_hadamard1_out // nb_tiles_A
        len_hadamard1_packed_tile = len_hadamard1_packed // nb_tiles_A
        len_hadamard1_packed_tile_re = len_hadamard1_packed_tile // 2
        len_phaseA_dma_per_tile_per_part = dModel_tile * L * FP8 // 8
        # Phase C K-tile B slot sizes.
        len_weight3_ktile = len_weight3 // nb_tiles_C
        len_hadamard2_packed_ktile = len_hadamard2_packed // nb_tiles_C

        specs = [
            ("weight1", len_weight1),
            ("weight2", len_weight2),
            ("weight3", len_weight3),
            ("in", len_in),
            ("partition1_out", len_partition1_out),
            ("twiddles1", len_twiddles1),
            ("hadamard1_out", len_hadamard1_out),
            ("hadamard1_packed", len_hadamard1_packed),
            ("partition2_out", len_partition2_out),
            ("twiddles2", len_twiddles2),
            ("hadamard2_out", len_hadamard2_out),
            ("hadamard2_packed", len_hadamard2_packed),
            ("partition3_out", len_partition3_out),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)

        tile_scalars = {
            "dModel_tile": dModel_tile,
            "N_1_tile": N_1_tile,
            "length_in_tile": len_in_tile,
            "length_partition1_out_tile": len_partition1_out_tile,
            "length_hadamard1_out_tile": len_hadamard1_out_tile,
            "length_hadamard1_packed_tile": len_hadamard1_packed_tile,
            "length_hadamard1_packed_tile_re": len_hadamard1_packed_tile_re,
            "phaseA_dma_per_tile_per_part": len_phaseA_dma_per_tile_per_part,
            "dInner_3_tile": dInner_3_tile,
            "length_weight3_ktile": len_weight3_ktile,
            "length_hadamard2_packed_ktile": len_hadamard2_packed_ktile,
        }
        scalars = {**lengths, **deltas, **tile_scalars}

        test_data = {
            **{
                name: "uint8_t"
                for name in (
                    "dft_weight1",
                    "dft_weight2",
                    "dft_weight3",
                    "dft_in",
                    "partition1_expected",
                    "twiddles1",
                    "hadamard1_expected",
                    "partition2_expected",
                    "twiddles2",
                    "hadamard2_expected",
                    "partition3_expected",
                )
            }
        }
        tests = {"expected": 2 * L * dModel}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
