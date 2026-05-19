#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# 3-way partitioned EinFFT (L = L1 * L2 * L3).
#
# Pipeline:
#   step 1 : partition1  (IS-core ISGEMM_SQ_TRANSPOSE)  W_L1 @ stack_L1
#   step 2 : hadamard1   (SIMD CMUL FP8) with twiddles1 broadcast across dModel
#   step 2B: reorder1    (SIMD NOOP FP8) — deinterleaves re/im (same pattern as 2-way fft)
#   step 2C: dma_gather1 (DMA engine) — strided byte gather, __flattenColMajor
#            -> partition2 B layout (buildStackedInput_L2_3way + flattenB).
#   step 3 : partition2  (IS-core ISGEMM_SQ)            W_L2 @ stack_L2_3way
#   step 4 : hadamard2   (SIMD CMUL FP8) with twiddles2 broadcast across L1 and dModel
#   step 4B: reorder2    (SIMD NOOP FP8) — deinterleaves re/im
#   step 4C: dma_gather2 (DMA engine) — strided byte gather, __flattenColMajor
#            -> partition3 B layout (buildStackedInput_L3_3way + flattenB).
#   step 5 : partition3  (IS-core ISGEMM_SQ)            W_L3 @ stack_L3_3way
#
# WHY the DMA gather (instead of a single SIMD-NOOP reorder pass, like 2-way fft):
#   The 2-way fft's reorder works because __flattenColMajor over (l = k1*L2 + m2)
#   coincidentally equals buildStackedInput_L2 + flattenB when L2 == seqLenUnroll.
#   In 3-way the row order is l = k1*L2*L3 + m2*L3 + m3 (m3 innermost), but
#   partition2's B layout needs m2 innermost — an m2 <-> m3 byte transpose.
#   R7's 2x2 spatial x 8-bytes-per-bank delivery can't gather 16 stride-L3 bytes
#   in one cycle, and W3 has the symmetric scatter problem. A multi-pass SIMD-NOOP
#   choreography to express this transpose is doable but non-trivial; until that's
#   designed, the DMA engine does the gather (size=1, src_stride=L3, repeat=L2 per
#   chunk; 2*L1*L3*D chunks total), still hardware-streaming and no Snitch scalar
#   loops in the critical path.
#
# Differences from the 2-way generator:
#   - partition1 output has shape (2*L1, L2*L3*D)   -> N_1 = dModel * L2 * L3
#   - partition2 output has shape (2*L2, L1*L3*D)   -> N_2 = dModel * L1 * L3
#   - partition3 output has shape (2*L3, L1*L2*D)   -> N_3 = dModel * L1 * L2
#   - twiddles1 has L entries indexed by l   (computeTwiddle(L1, L2*L3) flat)
#   - twiddles2 has L2*L3 entries indexed by (k2*L3+m3) and is broadcast across L1 and D

import pathlib
import sys
import os

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import DataGeneratorBase, FP8, BF16, BANK_BYTES  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "fft-3way"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)

    def run(self):
        self.build_data()

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

        # GEMM dims for each partition
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
        assert L1 % seqLenUnroll == 0, f"L1 ({L1}) must be a multiple of {seqLenUnroll}"
        assert L1_padded // dInnerUnroll == L1 // seqLenUnroll
        assert (2 * suc_serial_width_BC) // 8 == 4 * BANK_BYTES, "SIMD input width must be 4 banks"

        # Downsizer accounting (matches fft 2-way)
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = seqLenUnroll * FP8
        b_downsize_factor = b_in_width / b_array_width  # >= 1
        downsized_N_1 = int(N_1 / b_downsize_factor)
        downsized_N_2 = int(N_2 / b_downsize_factor)
        downsized_N_3 = int(N_3 / b_downsize_factor)
        assert downsized_N_1 * b_in_width == N_1 * b_array_width
        assert downsized_N_2 * b_in_width == N_2 * b_array_width
        assert downsized_N_3 * b_in_width == N_3 * b_array_width

        cd_array_width = seqLenUnroll * BF16
        assert cd_array_width == b_in_width, "Memory layout mismatch"

        psum_bounds_and_strides_1 = ([M_1 * N_1, K_1], [cd_array_width // 8, 0])
        psum_bounds_and_strides_2 = ([M_2 * N_2, K_2], [cd_array_width // 8, 0])
        psum_bounds_and_strides_3 = ([M_3 * N_3, K_3], [cd_array_width // 8, 0])

        streamers = {
            #
            # Step 1: partition 1   (W_L1: 2*L1 x L1_padded,   stack_L1: L1_padded x L2*L3*D)
            #
            "R11_1": (
                [2 * L1 * L1_padded * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12_1": (
                [downsized_N_1, M_1, K_1],
                [b_in_width // 8, 0, downsized_N_1 * b_in_width // 8],
            ),
            "R13_1": psum_bounds_and_strides_1,
            "W3_1": psum_bounds_and_strides_1,
            #
            # Step 2: hadamard 1 (CMul). Reads partition1_out (FP8 in BF16 slots, 2*L1 x L2*L3*D,
            # bank-transposed by partition1's ISGEMM_SQ_TRANSPOSE).
            #
            # W3_2 writes 32 bytes/cycle sequentially, so cycle c must produce CMul results for
            # flat indices [c*16, c*16+15] in __flattenColMajor (d outer, l inner). For 3-way
            # the row order is l = k1*L2*L3 + m2*L3 + m3 (m3 inner stride 1). For 16 sequential
            # l values per cycle to match m3=0..15 within one (k1, m2, d), the cycle order
            # must be: m2 innermost, k1 middle, d outer.
            #
            # Per cycle the SIMD core processes 16 N values for one M_inner. In the bank-
            # transposed M-tile, the 16 N values needed (=N = d*L2*L3 + m2*L3 + 0..15) span
            # exactly 2 adjacent matrices, so R7's two reals slots (spatial[0]=L1*BANK_BYTES=128
            # = one matrix) cover them. The M_inner of those slots is determined by the
            # middle counter (k1 advancing through banks within a matrix-pair).
            #
            # Strides:
            #   inner (m2): N advances by L3 = 16, i.e. (L3/8) = 2 matrices, byte stride = 2*L1*BANK_BYTES
            #   middle (k1): byte stride = BANK_BYTES (= next bank within a matrix → next M_inner)
            #   outer (d): byte stride = L*FP8/8 (= next d's data)
            #
            # Per CMul cycle the SIMD core processes 16 m3 values for one (k1, m2, d).
            # When L1 == seqLenUnroll (16), k1 fits in one M-tile → 3-level (m2, k1, d).
            # When L1 > seqLenUnroll, we need a 4th level for the M-tile index within reals.
            #
            # cycle (inner -> outer): m2, k1_low (= M_inner), k1_high (= M-tile-within-reals), d.
            "R7_2": (
                [L2, seqLenUnroll, L1 // seqLenUnroll, dModel],
                [
                    (L3 * FP8 // 8 // BANK_BYTES) * seqLenUnroll * BANK_BYTES,  # m2 → +(L3/8) matrices
                    BANK_BYTES,                                                  # k1_low → +1 bank
                    seqLenUnroll * L2 * L3 * dModel * FP8 // 8,                  # k1_high → +1 M-tile (only fires if L1 > 16)
                    L * FP8 // 8,                                                # d → +1 d's data
                ],
                [seqLenUnroll * BANK_BYTES, L * dModel * FP8 // 8],
            ),
            # Twiddles1 group index = k1*(L_inner/16) + m2*(L3/16). For L3=16: group = k1*L2 + m2.
            # m2 inner → group +=1 → stride 32; k1_low → group += L2 → stride L2*32;
            # k1_high → group += seqLenUnroll * L2 → stride seqLenUnroll*L2*32 (only fires for L1>16).
            "R13_2": (
                [L2, seqLenUnroll, L1 // seqLenUnroll, dModel],
                [4 * BANK_BYTES, L2 * 4 * BANK_BYTES, seqLenUnroll * L2 * 4 * BANK_BYTES, 0],
            ),
            "W3_2": (
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            #
            # Step 2B: deinterleave re/im in hadamard1_out -> hadamard1_packed (= __flattenColMajor
            # layout of (L,D) reals followed by imags). NOT yet partition2's B layout; a scalar
            # reshape (step 2C in fft.c) gathers the m2 axis to produce hadamard1_reordered.
            #
            "R7_2B": (
                [L * dModel * FP8 // (2 * suc_serial_width_BC), 2],
                [8 * BANK_BYTES, 2 * BANK_BYTES],
                [BANK_BYTES, 4 * BANK_BYTES],
            ),
            "W3_2B": (
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            #
            # Step 3: partition 2   (W_L2: 2*L2 x 2*L2_padded,   B: 2*L2_padded x L1*L3*D)
            #
            "R11_3": (
                [2 * L2 * 2 * L2_padded * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12_3": (
                [downsized_N_2, M_2, K_2],
                [b_in_width // 8, 0, downsized_N_2 * b_in_width // 8],
            ),
            "R13_3": psum_bounds_and_strides_2,
            "W3_3": psum_bounds_and_strides_2,
            #
            # Step 4: hadamard 2 (CMul with twiddles2). Reads partition2_out (FP8 in BF16 slots,
            # 2*L2 x L1*L3*D, bank-transposed by partition2's ISGEMM_SQ_TRANSPOSE).
            #
            # Same cycle structure as step 2 but with the m2 axis replaced by k2 (= IS-core M_inner)
            # and the inner-N axis now m3. With buildStackedInput_L2_3way's col arrangement
            # (d outer, k1 middle, m3 inner), 16 consecutive m3 values are contiguous in N → 2
            # adjacent matrices in the banked layout → same R7 gather as R7_2.
            #
            # cycle order (inner -> outer): k2 (= M_inner of partition2 output bank-transpose),
            #                                k1 (= matrix-pair within d), d.
            "R7_4": (
                [L2, L1, dModel, 1],
                [BANK_BYTES, (L3 // (BANK_BYTES // (FP8 // 8))) * L2 * BANK_BYTES, L * FP8 // 8, 0],
                [L2 * BANK_BYTES, L * dModel * FP8 // 8],
            ),
            # Twiddles2 = computeTwiddle(L2, L3) flat — shape (L2, L3) so (k2, m3) ordering.
            # SIMD-interleaved groups of 16: group g = (k2 fixed, m3 = g*16..g*16+15 mod L3, ...).
            # For L3 = 16 each group is exactly one k2's 16 m3 values, so group index = k2.
            # In our (k2 inner, k1 middle, d outer) cycle order:
            #   inner k2 → group += 1   → byte stride = 4*BANK_BYTES
            #   middle k1 → re-read same table → stride 0
            #   outer d   → re-read → stride 0
            "R13_4": (
                [L2, L1, dModel, 1],
                [4 * BANK_BYTES, 0, 0, 0],
            ),
            "W3_4": (
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            #
            # Step 4B: deinterleave hadamard2_out -> hadamard2_packed (__flattenColMajor).
            #
            "R7_4B": (
                [L * dModel * FP8 // (2 * suc_serial_width_BC), 2],
                [8 * BANK_BYTES, 2 * BANK_BYTES],
                [BANK_BYTES, 4 * BANK_BYTES],
            ),
            "W3_4B": (
                [2 * L * dModel * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            #
            # Step 5: partition 3   (W_L3: 2*L3 x 2*L3_padded,   B: 2*L3_padded x L1*L2*D)
            #
            "R11_5": (
                [2 * L3 * 2 * L3_padded * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12_5": (
                [downsized_N_3, M_3, K_3],
                [b_in_width // 8, 0, downsized_N_3 * b_in_width // 8],
            ),
            "R13_5": psum_bounds_and_strides_3,
            "W3_5": psum_bounds_and_strides_3,
        }

        specs = [
            ("weight1", 2 * L1 * L1_padded * FP8 // 8),
            ("weight2", 2 * L2 * 2 * L2_padded * FP8 // 8),
            ("weight3", 2 * L3 * 2 * L3_padded * FP8 // 8),
            ("in", L * dModel * FP8 // 8),
            ("partition1_out", 2 * L * dModel * BF16 // 8),
            ("twiddles1", 2 * L * FP8 // 8),
            ("hadamard1_out", 2 * L * dModel * FP8 // 8),
            # `hadamard1_packed`: SIMD-NOOP deinterleaved layout (__flattenColMajor
            # [reals | imags]) with the chip's NEW byte order (m2 stride-1 in chunks).
            # Partition2 reads it directly via R12 — no inter-stage reorder pass.
            ("hadamard1_packed", 2 * L * dModel * FP8 // 8),
            ("partition2_out", 2 * L * dModel * BF16 // 8),
            ("twiddles2", 2 * L2 * L3 * FP8 // 8),
            ("hadamard2_out", 2 * L * dModel * FP8 // 8),
            # `hadamard2_packed`: same role as `hadamard1_packed` but post-stage-2.
            # Partition3 reads it directly via R12 (m3 stride-1 in chunks).
            ("hadamard2_packed", 2 * L * dModel * FP8 // 8),
            ("partition3_out", 2 * L * dModel * BF16 // 8),
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
