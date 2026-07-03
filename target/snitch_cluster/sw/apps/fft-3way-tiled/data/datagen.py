#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Tiled 3-way partitioned EinFFT (L = L1 * L2 * L3). See docs/dataflow/05_fft.md

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


def align64(value: int) -> int:
    return (value + 63) & ~63


class DataGenerator(DataGeneratorBase):
    APP_NAME = "fft-3way-tiled"

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
        spec = importlib.util.spec_from_file_location("memory_model_fft3", os.path.join(app_dir, "memory_model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from memory_model_base import run_model_from_datagen  # type: ignore[import]

        self.lines_params.append(run_model_from_datagen(mod.build_report, app_dir))

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
        nb_d = self.kwargs["nb_tiles_A"]  # outer dModel-slice count
        # l3-tiling: stages 1-4 run per l3-tile; partition3 K-accumulates the nb_l3 tiles.
        l3_tile = self.kwargs.get("l3_tile", L3)
        assert L3 % l3_tile == 0, f"L3 ({L3}) must be divisible by l3_tile ({l3_tile})"
        nb_l3 = L3 // l3_tile
        # Whole partition-3 psum stays on-chip here, so output N-tiling is unsupported: pin
        # nb_ntile=1 so the partition-3 descriptors match the full-N GEMM.
        nb_ntile = 1

        # dModel is the independent batch axis of every partition: slice it and run
        # the full kernel per slice. dM = channels per slice.
        assert dModel % nb_d == 0, f"dModel ({dModel}) must be divisible by nb_tiles_A ({nb_d})"
        dM = dModel // nb_d

        # GEMM dims per partition. M/K are per-row/per-contraction (dModel-independent);
        # only the N (output) axis carries the (sliced) dModel factor.
        M_1 = (2 * L1) // seqLenUnroll
        M_2 = (2 * L2) // seqLenUnroll
        M_3 = (2 * L3) // seqLenUnroll
        K_1 = L1_padded // dInnerUnroll
        K_2 = 2 * L2_padded // dInnerUnroll
        K_3 = 2 * L3_padded // dInnerUnroll
        # partition3 K-reduction split into nb_l3 invocations (accumulate in BF16, requant on last).
        assert K_3 % nb_l3 == 0, f"K_3 ({K_3}) must be divisible by nb_l3 ({nb_l3})"
        K_3t = K_3 // nb_l3
        # Stages 1-4 run on one l3-tile at a time, on contiguous tile-local buffers gathered by
        # DMA; descriptors are the un-tiled ones with L3 -> L3t, L -> Lt.
        assert l3_tile % seqLenUnroll == 0, f"l3_tile ({l3_tile}) must be a multiple of {seqLenUnroll}"
        L3t = l3_tile
        Lt = L1 * L2 * L3t
        assert L2 <= seqLenUnroll, f"L2 ({L2}) must be <= seqLenUnroll ({seqLenUnroll}); see note above"
        N_1 = dM * L2 * L3t
        N_2 = dM * L1 * L3t
        N_3 = dM * L1 * L2
        assert L1 * L2 * L3 == L
        assert L1 % seqLenUnroll == 0, f"L1 ({L1}) must be a multiple of {seqLenUnroll}"
        assert L1_padded // dInnerUnroll == L1 // seqLenUnroll
        assert (2 * suc_serial_width_BC) // 8 == 4 * BANK_BYTES, "SIMD input width must be 4 banks"

        # Downsizer accounting.
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = seqLenUnroll * FP8
        b_downsize_factor = b_in_width / b_array_width  # >= 1
        downsized_N_1 = int(N_1 / b_downsize_factor)
        downsized_N_2 = int(N_2 / b_downsize_factor)
        downsized_N_3 = int(N_3 / b_downsize_factor)
        assert downsized_N_1 * b_in_width == N_1 * b_array_width
        assert downsized_N_2 * b_in_width == N_2 * b_array_width
        assert downsized_N_3 * b_in_width == N_3 * b_array_width
        # partition3 N-tile (output batch chunk)
        assert N_3 % nb_ntile == 0, f"N_3 ({N_3}) must be divisible by nb_ntile ({nb_ntile})"
        assert downsized_N_3 % nb_ntile == 0, f"downsized_N_3 ({downsized_N_3}) must be divisible by nb_ntile"
        N_3t = N_3 // nb_ntile
        downsized_N_3t = downsized_N_3 // nb_ntile

        cd_array_width = seqLenUnroll * BF16
        assert cd_array_width == b_in_width, "Memory layout mismatch"

        psum_bounds_and_strides_1 = ([M_1 * N_1, K_1], [cd_array_width // 8, 0])
        psum_bounds_and_strides_2 = ([M_2 * N_2, K_2], [cd_array_width // 8, 0])
        # partition3 psum walks ONE N-tile's output (M_3*N_3t), K_3t contraction steps per
        # invocation; the nb_l3 invocations read-accumulate into the same psum (stride-0 K).
        psum_bounds_and_strides_3 = ([M_3 * N_3t, K_3t], [cd_array_width // 8, 0])

        # Per-slice descriptors: identical to un-tiled fft-3way with `dModel -> dM`.
        streamers = {
            # Step 1: partition 1.
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
            # Step 2: hadamard 1 (CMul). Cycle order (inner→outer): m2, k1_low, k1_high, d.
            "R7_2": (
                [L2, seqLenUnroll, L1 // seqLenUnroll, dM],
                [
                    (L3t * FP8 // 8 // BANK_BYTES) * seqLenUnroll * BANK_BYTES,
                    BANK_BYTES,
                    seqLenUnroll * L2 * L3t * dM * FP8 // 8,
                    # d-slice within one bank-transposed M-tile (= Lt only when L1 == seqLenUnroll).
                    seqLenUnroll * L2 * L3t * FP8 // 8,
                ],
                [seqLenUnroll * BANK_BYTES, Lt * dM * FP8 // 8],
            ),
            "R13_2": (
                [L2, seqLenUnroll, L1 // seqLenUnroll, dM],
                [4 * BANK_BYTES, L2 * 4 * BANK_BYTES, seqLenUnroll * L2 * 4 * BANK_BYTES, 0],
            ),
            "W3_2": (
                [2 * Lt * dM * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 2B: SIMD NOOP deinterleave re/im → __flattenColMajor.
            "R7_2B": (
                [Lt * dM * FP8 // (2 * suc_serial_width_BC), 2],
                [8 * BANK_BYTES, 2 * BANK_BYTES],
                [BANK_BYTES, 4 * BANK_BYTES],
            ),
            "W3_2B": (
                [2 * Lt * dM * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 3: partition 2.
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
            # Step 4: hadamard 2 (CMul with twiddles2). Cycle order (inner→outer): k2_low, k2_high,
            # k1, d.
            "R7_4": (
                [seqLenUnroll, L2 // seqLenUnroll, L1, dM],
                [
                    BANK_BYTES,                                          # k2_low: next m1 within m2-group
                    L1 * L3t * dM * seqLenUnroll * FP8 // 8,             # k2_high: next m2-group (16 k2)
                    (L3t // (BANK_BYTES // (FP8 // 8))) * seqLenUnroll * BANK_BYTES,  # k1: += L3 cols
                    L1 * L3t * seqLenUnroll * FP8 // 8,                  # d: += L1*L3 cols (dM bound-1)
                ],
                [seqLenUnroll * BANK_BYTES, Lt * dM * FP8 // 8],         # m3 group-of-8 jump (matrix), re/im
            ),
            "R13_4": (
                [seqLenUnroll, L2 // seqLenUnroll, L1, dM],
                [4 * BANK_BYTES, seqLenUnroll * 4 * BANK_BYTES, 0, 0],
            ),
            "W3_4": (
                [2 * Lt * dM * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 4B: SIMD NOOP deinterleave re/im → __flattenColMajor.
            "R7_4B": (
                [Lt * dM * FP8 // (2 * suc_serial_width_BC), 2],
                [8 * BANK_BYTES, 2 * BANK_BYTES],
                [BANK_BYTES, 4 * BANK_BYTES],
            ),
            "W3_4B": (
                [2 * Lt * dM * FP8 // (2 * suc_serial_width_BC)],
                [4 * BANK_BYTES],
            ),
            # Step 5: partition 3. K-tile the 2*L3 contraction into nb_l3 chunks of K_3t from the
            # assembled [re|im] H2 and accumulate (requant on the last).
            "R11_5": (
                [2 * L3 * 2 * L3_padded * FP8 // iscore_serial_width // nb_l3],
                [iscore_serial_width // 8],
            ),
            "R12_5": (
                [downsized_N_3t, M_3, K_3t],
                [b_in_width // 8, 0, downsized_N_3t * b_in_width // 8],
            ),
            "R13_5": psum_bounds_and_strides_3,
            "W3_5": psum_bounds_and_strides_3,
        }
        # partition3 K-chunk base-pointer offsets (contiguous in the [re|im] order).
        weight3_kchunk_bytes = (2 * L3 * 2 * L3_padded * FP8 // 8) // nb_l3
        h2_kchunk_bytes = K_3t * downsized_N_3t * b_in_width // 8

        # H2 assembly offsets: each l3-tile's [re|im] halves go to the full H2's re/im regions.
        h2_ktile_bytes = K_3t * downsized_N_3 * b_in_width // 8   # one l3-tile's H2 (re+im)
        h2_half_bytes = h2_ktile_bytes // 2                       # one re or im K-block
        h2_im_region = (2 * L * dM * FP8 // 8) // 2               # start of the im region in the full H2

        # ---------- Buffer sizes ------------------------------------------------
        # Weights/twiddles depend on L only (broadcast over d) → FULL, loaded once.
        len_weight1 = 2 * L1 * L1_padded * FP8 // 8
        len_weight2 = 2 * L2 * 2 * L2_padded * FP8 // 8
        len_weight3 = 2 * L3 * 2 * L3_padded * FP8 // 8
        len_twiddles1 = 2 * L * FP8 // 8
        len_twiddles2 = 2 * L2 * L3 * FP8 // 8
        # Activations: FULL is the golden/DMA size; SLICE is what lives in TCDM.
        len_in = L * dModel * FP8 // 8
        len_in_slice = L * dM * FP8 // 8
        len_partition3_out = 2 * L * dModel * BF16 // 8

        # Output scatter: M_3 FP8 row-blocks, slice s owns dM contiguous channels per block.
        # Block sizes use the real (FP8) row-block bytes, not the padded buffer length.
        out_block_full = dModel * L1 * L2 * seqLenUnroll * FP8 // 8
        out_block_slice = dM * L1 * L2 * seqLenUnroll * FP8 // 8
        out_nblk = M_3

        # Input gather. dft_in is K-tile-major ([k1_tile][d][L2*L3][seqLenUnroll]), so a slice's
        # dM channels sit at a per-K-tile offset; gather them with a 2-D transfer (repeat =
        # in_ktile_count, src_stride = one full N-sweep).
        in_ktile_count = L1 // seqLenUnroll
        in_slice_chunk = dM * L2 * L3 * seqLenUnroll * FP8 // 8  # one K-tile of a slice
        in_ktile_stride = dModel * L2 * L3 * seqLenUnroll * FP8 // 8  # one full N-sweep in L3

        # --- l3-tile gather of input + twiddles into contiguous tile-local buffers (from DRAM) ---
        # in_tile: per k1-tile, per d, copy the tile's m3-block (L3t*L2*Mu) out of the [d][L2*L3][Mu]
        # chip layout. Loop k1-tiles in C; each is a 2-D DMA over d.
        in_gather_chunk = L3t * L2 * seqLenUnroll * FP8 // 8      # one d's m3-block in one k1-tile
        in_gather_d_stride = L2 * L3 * seqLenUnroll * FP8 // 8    # full-L3 d stride in dft_in
        in_gather_dst_ktile = dM * L3t * L2 * seqLenUnroll * FP8 // 8  # one k1-tile in in_tile
        in_tile_bytes = len_in_slice // nb_l3                     # Lt*dM
        # twiddles1 (k1, j=m3*L2+m2), 2 bytes/entry (cos,sin): gather the tile's m3 j-block per k1.
        tw1_entry = 2 * FP8 // 8
        tw1_gather_chunk = L3t * L2 * tw1_entry
        tw1_gather_src_stride = L2 * L3 * tw1_entry
        tw1_tile_bytes = L1 * L3t * L2 * tw1_entry               # 2*Lt
        # twiddles2 (l2, l3), 2 bytes/entry: gather the tile's L3t columns per l2 row.
        tw2_gather_chunk = L3t * tw1_entry
        tw2_gather_src_stride = L3 * tw1_entry
        tw2_tile_bytes = L2 * L3t * tw1_entry

        # Tile-local psum/scratch (1/nb_l3 of the full slot).
        slot_size = align64(2 * L * dM * BF16 // 8)              # full partition3 psum (gemm3 out)
        slot_size_tile = align64(2 * Lt * dM * BF16 // 8)        # gemm1/2 psum per l3-tile
        hsize_tile = slot_size_tile // 2                          # FP8 cmul/noop scratch per tile

        specs = [
            ("weight1", len_weight1),
            ("weight2", len_weight2),
            ("weight3", len_weight3),
            ("in", len_in),
            ("twiddles1", len_twiddles1),
            ("twiddles2", len_twiddles2),
            ("partition3_out", len_partition3_out),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)

        slice_scalars = {
            "nb_d": nb_d,
            "dModel_slice": dM,
            "length_in_slice": len_in_slice,
            "in_ktile_count": in_ktile_count,
            "in_slice_chunk": in_slice_chunk,
            "in_ktile_stride": in_ktile_stride,
            "out_block_full": out_block_full,
            "out_block_slice": out_block_slice,
            "out_nblk": out_nblk,
            "slot_size": slot_size,
            "nb_l3": nb_l3,
            "l3_tile": l3_tile,
            # l3-tile gather + tile-local buffer sizes
            "in_gather_chunk": in_gather_chunk,
            "in_gather_d_stride": in_gather_d_stride,
            "in_gather_dst_ktile": in_gather_dst_ktile,
            "in_tile_bytes": in_tile_bytes,
            "tw1_gather_chunk": tw1_gather_chunk,
            "tw1_gather_src_stride": tw1_gather_src_stride,
            "tw1_tile_bytes": tw1_tile_bytes,
            "tw2_gather_chunk": tw2_gather_chunk,
            "tw2_gather_src_stride": tw2_gather_src_stride,
            "tw2_tile_bytes": tw2_tile_bytes,
            "slot_size_tile": slot_size_tile,
            "hsize_tile": hsize_tile,
            # partition3 K-tiling over l3-tiles + the [re|im] H2 assembly offsets
            "h2_half_bytes": h2_half_bytes,                    # one l3-tile's re (or im) K-block
            "h2_im_region": h2_im_region,                      # start of the im region in the full H2
            "h2_full_bytes": 2 * h2_im_region,                 # full assembled H2 (re + im regions)
            "weight3_kchunk_bytes": weight3_kchunk_bytes,      # partition3 K-chunk weight offset
            "h2_kchunk_bytes": h2_kchunk_bytes,                # partition3 K-chunk h2 offset
        }
        scalars = {**lengths, **deltas, **slice_scalars}

        # TCDM peak + OOM guard: see memory_model.py (run from run() via _run_memory_model).

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
