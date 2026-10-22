#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Tiled FFT datagen. Keeps TCDM peak low by holding all large tensors in L3 and
# only the per-tile working slots + small weight/twiddle tables in TCDM.
# Design (tiling axes, buffer lifecycles, the flattenB L3 spill):
# docs/dataflow/05_fft.md, "fft-tiled" section.

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
    APP_NAME = "fft-tiled"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        local_params_path = self.params_in_path(__file__)
        with local_params_path.open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.build_data()
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
        L1_padded = self.kwargs["L1_padded"]
        L2_padded = self.kwargs["L2_padded"]
        nb_tiles = self.kwargs["nb_tiles"]
        nb_tiles_B = self.kwargs.get("nb_tiles_B", nb_tiles)

        # Tile dims — Phase A uses nb_tiles, Phase B uses nb_tiles_B (decoupled
        # because K_2 is small and limits the Phase B tile count independently).
        assert L2 % nb_tiles == 0, f"L2 ({L2}) must be divisible by nb_tiles ({nb_tiles})"
        assert dModel % nb_tiles == 0, f"dModel ({dModel}) must be divisible by nb_tiles ({nb_tiles})"
        L2_tile = L2 // nb_tiles  # Phase A tile size along L2
        dModel_tile = dModel // nb_tiles  # Phase A tile size along dModel (for SIMD)
        L_tile_a = L1 * L2_tile  # Phase A per-tile sequence length

        M_1 = (2 * L1) // seqLenUnroll
        M_2 = (2 * L2) // seqLenUnroll
        K_1 = L1_padded // dInnerUnroll
        K_2 = 2 * L2_padded // dInnerUnroll
        assert K_2 % nb_tiles_B == 0, f"K_2 ({K_2}) must be divisible by nb_tiles_B ({nb_tiles_B})"
        K_2_t = K_2 // nb_tiles_B  # K macros per Phase B tile invocation
        dInner_2_tile = K_2_t * dInnerUnroll  # K elements per Phase B tile invocation
        N_1 = dModel * L2
        N_2 = dModel * L1
        N_1_tile = dModel * L2_tile  # Phase A: GEMM N axis per tile

        assert L1 % seqLenUnroll == 0
        assert L1 * L2 == L
        assert L1_padded // dInnerUnroll == L1 // seqLenUnroll
        assert (2 * suc_serial_width_BC) // 8 == 4 * BANK_BYTES

        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = seqLenUnroll * FP8
        b_downsize_factor = b_in_width / b_array_width
        downsized_N_2 = int(N_2 / b_downsize_factor)
        downsized_N_1_tile = int(N_1_tile / b_downsize_factor)
        assert downsized_N_1_tile == N_1_tile / b_downsize_factor

        cd_array_width = seqLenUnroll * BF16
        assert cd_array_width == b_in_width

        # ---- Streamer bounds (per tile) ----
        # Phase A: N-axis tiled (dModel) — psum_p1 uses N_1_tile.
        # Phase B: K-axis tiled (dInner) — psum_p2 uses FULL M*N output, K_2_t outer K per tile.
        psum_p1_tile = ([M_1 * N_1_tile, K_1], [cd_array_width // 8, 0])
        psum_p2_ktile = ([M_2 * N_2, K_2_t], [cd_array_width // 8, 0])

        streamers = {
            #
            # Step 1: partition 1 (Phase A, tiled along L2)
            #
            "R11_1": (  # iscore A: weight1 (FULL, address fixed across tiles)
                [2 * L1 * L1_padded * FP8 // iscore_serial_width],
                [iscore_serial_width // 8],
            ),
            "R12_1": (  # iscore B: in tile slot in TCDM (= length_in_tile bytes)
                [downsized_N_1_tile, M_1, K_1],
                [
                    b_in_width // 8,
                    0,
                    downsized_N_1_tile * b_in_width // 8,  # K_1=1 typically; harmless if K=1
                ],
            ),
            "R13_1": psum_p1_tile,
            "W3_1": psum_p1_tile,
            #
            # Step 2: hadamard (Phase A, tiled). Operates on per-tile L_tile_a sequence positions.
            #
            "R7_2": (  # SIMD input from per-tile partition1_out tile slot.
                # 2x2-spatial reader: spatial[0] picks the 2 banktranspose matrices, spatial[1]
                # the re/im split; temporal dims (inner->outer) are l2-high, l1_dft, M-tile, d
                # (per-dim comments below). The l2-high dim is bound 1 for L2 == seqLenUnroll
                # but required for L2 > seqLenUnroll, else R7_2 under-reads and the SIMD hangs.
                [
                    L2 // seqLenUnroll,  # l2-high: 16-wide l2 groups beyond the spatial vector
                    seqLenUnroll,  # l1_dft within a macro-row
                    L1 // seqLenUnroll,  # M-tile (macro-row) of l1_dft
                    dModel_tile,  # d (dModel, tiled)
                ],
                [
                    2 * seqLenUnroll * BANK_BYTES,  # l2-high: skip the 2 spatial[0] matrices
                    BANK_BYTES,  # l1_dft within a macro-row
                    seqLenUnroll * N_1_tile * FP8 // 8,  # jump to next M-tile
                    seqLenUnroll * L2 * FP8 // 8,  # next d
                ],
                [
                    # spatial[0]: the 2 banks read in parallel are one seqLenUnroll-sized
                    # matrix apart, NOT L1 apart — these coincide only for L1==seqLenUnroll.
                    seqLenUnroll * BANK_BYTES,
                    L * dModel_tile * FP8 // 8,  # spatial[1]: re/im split (= tile re region size)
                ],
            ),
            "R13_2": (  # twiddles — un-tiled storage, broadcast across dModel_tile.
                # Twiddles only depend on (l1, l2), NOT d, so dModel tiling needs no slicing.
                [
                    2 * L * FP8 // (2 * suc_serial_width_BC),  # inner = un-tiled
                    dModel_tile,  # outer = halved
                ],
                [4 * BANK_BYTES, 0],
            ),
            "W3_2": (  # SIMD output: fused re/im split into the per-tile had_reord slot,
                # [tile_reals (contig) | tile_imags (contig)]. spatial[0] = re/im half-beat,
                # spatial[1] = tile-local re/im region offset. Per-tile DMA-out then
                # scatters into the L3 standard layout.
                [2 * L * dModel_tile * FP8 // (2 * suc_serial_width_BC)],
                [2 * BANK_BYTES],
                [BANK_BYTES, L * dModel_tile * FP8 // 8],
            ),
            #
            # Step 3: partition 2 (Phase B, K-AXIS tiled, accumulating in place).
            #
            "R11_3": (  # iscore A (weight2) — PER-TILE slice (K-major, contiguous per tile)
                [(2 * L2 * 2 * L2_padded * FP8 // iscore_serial_width) // nb_tiles_B],
                [iscore_serial_width // 8],
            ),
            "R12_3": (  # iscore B (hadamard_reordered) — PER-TILE K-slice
                [downsized_N_2, M_2, K_2_t],
                [
                    b_in_width // 8,
                    0,
                    downsized_N_2 * b_in_width // 8,  # K-macro stride (only fires if K_2_t > 1)
                ],
            ),
            "R13_3": psum_p2_ktile,  # FULL output, accumulates
            "W3_3": psum_p2_ktile,
        }

        # ---------- Buffer sizes ------------------------------------------------
        len_weight1 = 2 * L1 * L1_padded * FP8 // 8
        len_weight2 = 2 * L2 * 2 * L2_padded * FP8 // 8
        len_in = L * dModel * FP8 // 8
        len_twiddles = 2 * L * FP8 // 8
        len_partition1_out = 2 * L * dModel * BF16 // 8
        len_hadamard_out = 2 * L * dModel * FP8 // 8
        len_hadamard_reordered = 2 * L * dModel * FP8 // 8
        len_partition2_out = 2 * L * dModel * BF16 // 8

        for name, value in (
            ("in", len_in),
            ("partition1_out", len_partition1_out),
            ("hadamard_out", len_hadamard_out),
            ("hadamard_reordered", len_hadamard_reordered),
            ("partition2_out", len_partition2_out),
            ("twiddles", len_twiddles),
        ):
            assert value % nb_tiles == 0, f"length_{name} ({value}) not divisible by nb_tiles ({nb_tiles})"

        # Per-tile sizes
        # Phase A: dModel-axis tiled (uses nb_tiles)
        len_in_tile = len_in // nb_tiles
        len_partition1_out_tile = len_partition1_out // nb_tiles
        len_hadamard_reordered_tile = len_hadamard_reordered // nb_tiles  # tile total (re + im)
        len_hadamard_reordered_tile_re = len_hadamard_reordered_tile // 2  # tile reals chunk
        # Phase B: K-axis tiled (uses nb_tiles_B; FULL partition2_out, partial weight2 + hadamard_reordered)
        len_weight2_ktile = len_weight2 // nb_tiles_B  # per-K-tile A bytes
        len_hadamard_reordered_ktile = len_hadamard_reordered // nb_tiles_B  # per-K-tile B bytes

        # ---- Twiddles: no relayout for dModel tiling ----
        # The FFT twiddles only depend on (l1, l2), not on d. With dModel tiling,
        # all tiles use the SAME twiddle table — just load it un-tiled once.
        len_twiddles_per_tile_padded = len_twiddles  # = 512 B (un-tiled)
        len_twiddles_tiled_total = len_twiddles  # one shared copy in TCDM

        # ---- Phase A spill: scatter each d-tile's col-major blocks into flattenB L3. ----
        # `hadamard_reordered` in L3 is flattenB (K-tile-major) `[reals_fb | imags_fb]` —
        # the layout the partition-2 IS-core reads (see flattenHadamardReordered in
        # DataGeneratorFFT). Each seqLenUnroll-elem block reals[m*L2+k2*seqLenUnroll .. ][d]
        # goes to flattenB block (k2, d, m). This value is the per-tile per-part byte count
        # (it equals the contiguous 1D spill size used on the L2 == seqLenUnroll fast path).
        len_phaseA_dma_per_tile_per_part = dModel_tile * L * FP8 // 8  # = 4096 B for nb=2

        # Phase B is K-axis tiled (accumulating into FULL partition2_out in TCDM).
        # Per tile: DMA-in K-slice of hadamard_reordered into the TCDM tile B-slot;
        # update BASE_PTR_READER_11 to slice weight2 by K-macro offset. Runs once,
        # then verify partition2_out in TCDM (no DMA-out needed).

        specs = [
            ("weight1", len_weight1),
            ("weight2", len_weight2),
            ("in", len_in),
            ("partition1_out", len_partition1_out),
            ("twiddles", len_twiddles),
            ("hadamard_out", len_hadamard_out),
            ("hadamard_reordered", len_hadamard_reordered),
            ("partition2_out", len_partition2_out),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)

        tile_scalars = {
            "L2_tile": L2_tile,
            "dModel_tile": dModel_tile,
            "L_tile_a": L_tile_a,
            "N_1_tile": N_1_tile,
            "length_in_tile": len_in_tile,
            "length_partition1_out_tile": len_partition1_out_tile,
            "length_hadamard_reordered_tile": len_hadamard_reordered_tile,
            "length_hadamard_reordered_tile_re": len_hadamard_reordered_tile_re,
            # ---- twiddle (shared across tiles, no relayout) ----
            "length_twiddles_per_tile_padded": len_twiddles_per_tile_padded,
            "length_twiddles_tiled_total": len_twiddles_tiled_total,
            # ---- Phase A DMA-out (per-tile, per-part) ----
            "phaseA_dma_per_tile_per_part": len_phaseA_dma_per_tile_per_part,
            # ---- Phase B K-axis tiling ----
            "dInner_2_tile": dInner_2_tile,
            "length_weight2_ktile": len_weight2_ktile,
            "length_hadamard_reordered_ktile": len_hadamard_reordered_ktile,
            "nb_tiles_B": nb_tiles_B,
        }
        scalars = {**lengths, **deltas, **tile_scalars}

        test_data = {
            **{
                name: "uint8_t"
                for name in (
                    "dft_weight1",
                    "dft_weight2",
                    "dft_in",
                    "partition1_expected",
                    "twiddles",
                    "hadamard_expected",
                    "hadamard_reordered",
                    "partition2_expected",
                )
            }
        }
        tests = {"expected": 2 * L * dModel}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
