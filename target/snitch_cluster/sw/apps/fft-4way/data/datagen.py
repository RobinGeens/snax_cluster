#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# 4-way partitioned EinFFT (L = L1 * L2 * L3 * L4). Un-tiled (all buffers resident),
# used to bring up / validate the 4-partition kernel with per-stage golden checks.
# See docs/dataflow/05_fft.md.

import pathlib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import DataGeneratorBase, FP8, BF16, BANK_BYTES  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "fft-4way"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)

    def run(self):
        self.build_data()

    def build_data(self):
        mode_id = 6
        assert f"M{mode_id}_ISGEMM_SQ" in self.kwargs, f"verify mode_id {mode_id} for ISGEMM_SQ"
        slu = self.kwargs["seqLenUnroll"]
        diu = self.kwargs["dInnerUnroll"]
        iscore_serial_width = self.kwargs["iscore_serial_width"]
        suc_bc = self.kwargs["suc_serial_width_BC"]
        dModel = self.kwargs["dModel"]
        L = self.kwargs["seqLen"]
        L1 = self.kwargs["L1"]
        L2 = self.kwargs["L2"]
        L3 = self.kwargs["L3"]
        L4 = self.kwargs["L4"]
        L1p = self.kwargs["L1_padded"]
        L2p = self.kwargs["L2_padded"]
        L3p = self.kwargs["L3_padded"]
        L4p = self.kwargs["L4_padded"]
        L34 = L3 * L4

        # GEMM dims per partition.
        M_1 = (2 * L1) // slu
        M_2 = (2 * L2) // slu
        M_3 = (2 * L3) // slu
        M_4 = (2 * L4) // slu
        K_1 = L1p // diu
        K_2 = 2 * L2p // diu
        K_3 = 2 * L3p // diu
        K_4 = 2 * L4p // diu
        N_1 = dModel * L2 * L3 * L4
        N_2 = dModel * L1 * L3 * L4
        N_3 = dModel * L1 * L2 * L4
        N_4 = dModel * L1 * L2 * L3
        assert L1 * L2 * L3 * L4 == L
        assert L1 % slu == 0, f"L1 ({L1}) must be a multiple of {slu}"
        assert L1p // diu == L1 // slu
        assert (2 * suc_bc) // 8 == 4 * BANK_BYTES, "SIMD input width must be 4 banks"
        assert L2 <= slu, f"L2 ({L2}) must be <= seqLenUnroll ({slu})"

        # Downsizer accounting.
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = slu * FP8
        bdf = b_in_width / b_array_width  # >= 1
        dN_1 = int(N_1 / bdf)
        dN_2 = int(N_2 / bdf)
        dN_3 = int(N_3 / bdf)
        dN_4 = int(N_4 / bdf)
        for n, dn in ((N_1, dN_1), (N_2, dN_2), (N_3, dN_3), (N_4, dN_4)):
            assert dn * b_in_width == n * b_array_width

        cd_w = slu * BF16
        assert cd_w == b_in_width, "Memory layout mismatch"

        psum1 = ([M_1 * N_1, K_1], [cd_w // 8, 0])
        psum2 = ([M_2 * N_2, K_2], [cd_w // 8, 0])
        psum3 = ([M_3 * N_3, K_3], [cd_w // 8, 0])
        psum4 = ([M_4 * N_4, K_4], [cd_w // 8, 0])

        # cmul2 (twiddle2) trailing axis m34 = L3*L4 spans n_m34 SIMD vectors of 16 lanes.
        assert L34 % slu == 0, f"L3*L4 ({L34}) must be a multiple of {slu}"
        n_m34 = L34 // slu

        # Bank-transpose constants (FP8 in, BANKWIDTH=64): elemPerBank = 8, numBanks = 2,
        # matrix_bytes = slu*BANK. A SIMD CMul group reads 16 consecutive chip-columns at fixed
        # k1 from the bank-transposed partition output; the k1 index is split lo/hi by the
        # transpose (bank = k1 % numBanks, colGroup = k1 // numBanks). Assumes L1 == seqLenUnroll.
        elem_per_bank = BANK_BYTES
        num_banks = slu // elem_per_bank
        matrix_bytes = slu * BANK_BYTES
        assert L1 == slu, f"cmul descriptors derived for L1 == seqLenUnroll ({slu}); got L1={L1}"

        # cmul3 (twiddles3): partition3 row = k3, trailing axis m4 = L4. Two regimes.
        if L4 >= slu:
            # group = 16 consecutive m4 (=col3) at fixed (a=k1*L2+k2, k3); same form as cmul2.
            r7_6 = (
                [L4 // slu, L3, L1 * L2, 1],
                [num_banks * matrix_bytes, elem_per_bank, (L4 // elem_per_bank) * matrix_bytes, 0],
                [matrix_bytes, L3 * elem_per_bank],
            )
            r13_6 = ([L4 // slu, L3, L1 * L2, 1], [2 * slu, 2 * L4, 0, 0])
        else:
            # L4 < slu: a SIMD group spans (slu//L4) consecutive k3 rows x L4 m4 (bring-up, L4=8).
            kpg = slu // L4  # k3 per group
            r7_6 = (
                [L3 // kpg, L1 * L2, 1, 1],
                [kpg * elem_per_bank, (L4 // elem_per_bank) * matrix_bytes, 0, 0],
                [elem_per_bank, L3 * elem_per_bank],
            )
            r13_6 = ([L3 // kpg, L1 * L2, 1, 1], [2 * slu, 0, 0, 0])

        streamers = {
            # ---- Step 1: partition 1 ----
            "R11_1": ([2 * L1 * L1p * FP8 // iscore_serial_width], [iscore_serial_width // 8]),
            "R12_1": ([dN_1, M_1, K_1], [b_in_width // 8, 0, dN_1 * b_in_width // 8]),
            "R13_1": psum1,
            "W3_1": psum1,
            # ---- Step 2: hadamard 1 (CMul, chip layout). Cycle (inner->outer): chip_col_group
            # (incl d), k1_lo (=k1%numBanks), k1_hi (=k1//numBanks). ----
            "R7_2": (
                [L2 * L34 * dModel // slu, num_banks, L1 // num_banks, 1],
                [num_banks * matrix_bytes, elem_per_bank, num_banks * elem_per_bank, 0],
                [matrix_bytes, L * dModel * FP8 // 8],
            ),
            "R13_2": (
                [L2 * L34 * dModel // slu, L1, 1, 1],
                [2 * slu, 2 * L2 * L34, 0, 0],
            ),
            "W3_2": ([2 * L * dModel * FP8 // (2 * suc_bc)], [4 * BANK_BYTES]),
            # ---- Step 2B: reorder 1 (NOOP). Block-swap deinterleave (R7 ss=[2*BANK, BANK],
            # W3 sequential) into partition2's [re, im] K-tiles. ----
            "R7_2B": (
                [L * dModel // slu, 1, 1, 1],
                [4 * BANK_BYTES, 0, 0, 0],
                [2 * BANK_BYTES, BANK_BYTES],
            ),
            "W3_2B": ([2 * L * dModel * FP8 // (2 * suc_bc)], [4 * BANK_BYTES]),
            # ---- Step 3: partition 2 ----
            "R11_3": ([2 * L2 * 2 * L2p * FP8 // iscore_serial_width], [iscore_serial_width // 8]),
            "R12_3": ([dN_2, M_2, K_2], [b_in_width // 8, 0, dN_2 * b_in_width // 8]),
            "R13_3": psum2,
            "W3_3": psum2,
            # ---- Step 4: hadamard 2 (CMul, twiddles2). Cycle (inner->outer): m34_grp, k2, k1.
            # Assumes L34 >= slu. ----
            "R7_4": (
                [L34 // slu, L2, L1, 1],
                [num_banks * matrix_bytes, elem_per_bank, (L34 // elem_per_bank) * matrix_bytes, 0],
                [matrix_bytes, L2 * elem_per_bank],
            ),
            "R13_4": (
                [L34 // slu, L2, L1, 1],
                [2 * slu, 2 * L34, 0, 0],
            ),
            "W3_4": ([2 * L * dModel * FP8 // (2 * suc_bc)], [4 * BANK_BYTES]),
            # ---- Step 4B: reorder 2 (NOOP). Plain block-swap deinterleave (m3 already contiguous
            # under the contraction-order chip layout). ----
            "R7_4B": (
                [L * dModel // slu, 1, 1, 1],
                [4 * BANK_BYTES, 0, 0, 0],
                [2 * BANK_BYTES, BANK_BYTES],
            ),
            "W3_4B": ([2 * L * dModel * FP8 // (2 * suc_bc)], [4 * BANK_BYTES]),
            # ---- Step 5: partition 3 (bank-transposed) ----
            "R11_5": ([2 * L3 * 2 * L3p * FP8 // iscore_serial_width], [iscore_serial_width // 8]),
            "R12_5": ([dN_3, M_3, K_3], [b_in_width // 8, 0, dN_3 * b_in_width // 8]),
            "R13_5": psum3,
            "W3_5": psum3,
            # ---- Step 6: hadamard 3 (CMul, twiddles3). Group = m4 (+k3 if L4<slu) at fixed a;
            # r7_6/r13_6 computed above (two regimes for L4>=slu). ----
            "R7_6": r7_6,
            "R13_6": r13_6,
            "W3_6": ([2 * L * dModel * FP8 // (2 * suc_bc)], [4 * BANK_BYTES]),
            # ---- Step 6B: reorder 3 (NOOP). Block-swap deinterleave into partition4's
            # [re, im] K-tiles. ----
            "R7_6B": (
                [L * dModel // slu, 1, 1, 1],
                [4 * BANK_BYTES, 0, 0, 0],
                [2 * BANK_BYTES, BANK_BYTES],
            ),
            "W3_6B": ([2 * L * dModel * FP8 // (2 * suc_bc)], [4 * BANK_BYTES]),
            # ---- Step 7: partition 4 (final, plain) ----
            "R11_7": ([2 * L4 * 2 * L4p * FP8 // iscore_serial_width], [iscore_serial_width // 8]),
            "R12_7": ([dN_4, M_4, K_4], [b_in_width // 8, 0, dN_4 * b_in_width // 8]),
            "R13_7": psum4,
            "W3_7": psum4,
        }

        specs = [
            ("weight1", 2 * L1 * L1p * FP8 // 8),
            ("weight2", 2 * L2 * 2 * L2p * FP8 // 8),
            ("weight3", 2 * L3 * 2 * L3p * FP8 // 8),
            ("weight4", 2 * L4 * 2 * L4p * FP8 // 8),
            ("in", L * dModel * FP8 // 8),
            ("partition1_out", 2 * L * dModel * BF16 // 8),
            ("twiddles1", 2 * L * FP8 // 8),
            ("hadamard1_out", 2 * L * dModel * FP8 // 8),
            ("hadamard1_packed", 2 * L * dModel * FP8 // 8),
            ("partition2_out", 2 * L * dModel * BF16 // 8),
            ("twiddles2", 2 * L2 * L34 * FP8 // 8),
            ("hadamard2_out", 2 * L * dModel * FP8 // 8),
            ("hadamard2_packed", 2 * L * dModel * FP8 // 8),
            ("partition3_out", 2 * L * dModel * BF16 // 8),
            ("twiddles3", 2 * L3 * L4 * FP8 // 8),
            ("hadamard3_out", 2 * L * dModel * FP8 // 8),
            ("hadamard3_packed", 2 * L * dModel * FP8 // 8),
            ("partition4_out", 2 * L * dModel * BF16 // 8),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        scalars = {**lengths, **deltas}

        test_data = {
            **{
                name: "uint8_t"
                for name in (
                    "dft_weight1",
                    "dft_weight2",
                    "dft_weight3",
                    "dft_weight4",
                    "dft_in",
                    "partition1_expected",
                    "twiddles1",
                    "hadamard1_expected",
                    "partition2_expected",
                    "twiddles2",
                    "hadamard2_expected",
                    "partition3_expected",
                    "twiddles3",
                    "hadamard3_expected",
                    "partition4_expected",
                )
            }
        }
        tests = {"expected": 2 * L * dModel}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
