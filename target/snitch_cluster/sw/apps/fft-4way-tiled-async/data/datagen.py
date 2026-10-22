#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Datagen for the l3-streamed 4-way partitioned EinFFT (L = L1*L2*L3*L4). Reuses
# DataGeneratorFFT4Way (the un-tiled generator) and descriptors, with L3->L3t l3-tiling.
# See docs/dataflow/05_fft.md "fft-4way-tiled-async".

import pathlib
import sys
import os
import hjson

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import DataGeneratorBase, FP8, BF16, BANK_BYTES  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


def align64(value: int) -> int:
    return (value + 63) & ~63


class DataGenerator(DataGeneratorBase):
    APP_NAME = "fft-4way-tiled-async"

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
        spec = importlib.util.spec_from_file_location("memory_model_fft4a", os.path.join(app_dir, "memory_model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from memory_model_base import run_model_from_datagen  # type: ignore[import]

        self.lines_params.append(run_model_from_datagen(mod.build_report, app_dir))

    def build_data(self):
        mode_id = 6
        assert f"M{mode_id}_ISGEMM_SQ" in self.kwargs, f"verify mode_id {mode_id} for ISGEMM_SQ"
        slu = self.kwargs["seqLenUnroll"]
        diu = self.kwargs["dInnerUnroll"]
        iscore_serial_width = self.kwargs["iscore_serial_width"]
        suc_bc = self.kwargs["suc_serial_width_BC"]
        dModel = self.kwargs["dModel"]
        L = self.kwargs["seqLen"]
        L1, L2, L3, L4 = self.kwargs["L1"], self.kwargs["L2"], self.kwargs["L3"], self.kwargs["L4"]
        L1p, L2p, L3p, L4p = (self.kwargs[f"L{i}_padded"] for i in (1, 2, 3, 4))
        nb_d = self.kwargs["nb_tiles_A"]
        l3_tile = self.kwargs.get("l3_tile", L3)
        nb_ntile = self.kwargs.get("nb_ntile", 1)
        nb_m4 = self.kwargs.get("nb_m4", 1)

        assert L1 * L2 * L3 * L4 == L
        assert L1 == slu, f"descriptors assume L1 == seqLenUnroll ({slu})"
        assert L2 <= slu, f"L2 ({L2}) must be <= seqLenUnroll ({slu})"
        assert L3 % l3_tile == 0, f"L3 ({L3}) must be divisible by l3_tile ({l3_tile})"
        assert dModel % nb_d == 0
        assert (2 * suc_bc) // 8 == 4 * BANK_BYTES, "SIMD input width must be 4 banks"
        dM = dModel // nb_d
        nb_l3 = L3 // l3_tile
        L34 = L3 * L4
        # Stage-1-2 tiling: either m3-blocks (l3_tile < L3) or m4-blocks (nb_m4 > 1), not both.
        assert L4 % nb_m4 == 0, f"L4 ({L4}) must be divisible by nb_m4 ({nb_m4})"
        assert nb_l3 == 1 or nb_m4 == 1, "use either l3_tile or nb_m4 tiling, not both"
        assert nb_m4 == 1 or (L3 == slu and dM == 1), "nb_m4 tiling assumes L3 == seqLenUnroll and dM == 1"
        L3t = l3_tile
        L4t = L4 // nb_m4
        L34t = L3t * L4t
        Lt = L1 * L2 * L34t
        nb_12 = nb_l3 * nb_m4
        assert L34t % slu == 0, f"L3t*L4t ({L34t}) must be a multiple of {slu}"
        # Stage-3/4 N-tiling over the (k1,k2) column axis; packed3 stays resident.
        assert (L1 * L2) % nb_ntile == 0, f"L1*L2 must be divisible by nb_ntile ({nb_ntile})"
        assert nb_ntile == 1 or L4 == slu, "stage-3/4 N-tiling assumes the L4 == seqLenUnroll reorder3 regime"
        assert nb_ntile == 1 or nb_d == 1, "stage-3/4 N-tiling assumes nb_tiles_A == 1"

        elem_per_bank = BANK_BYTES
        num_banks = slu // elem_per_bank
        matrix_bytes = slu * BANK_BYTES
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = slu * FP8
        bdf = b_in_width / b_array_width
        cd_w = slu * BF16
        assert cd_w == b_in_width

        # GEMM dims. Stages 1-2 per-tile (Lt batch); stages 3-4 per (k1,k2) N-chunk.
        M_1, M_2, M_3, M_4 = (2 * L1) // slu, (2 * L2) // slu, (2 * L3) // slu, (2 * L4) // slu
        K_1, K_2, K_3, K_4 = L1p // diu, 2 * L2p // diu, 2 * L3p // diu, 2 * L4p // diu
        N_1, N_2 = dM * L2 * L34t, dM * L1 * L34t           # per stage-1-2 tile
        N_3, N_4 = dM * L1 * L2 * L4, dM * L1 * L2 * L3     # full
        A_c = L1 * L2 // nb_ntile
        N_3c, N_4c = N_3 // nb_ntile, N_4 // nb_ntile       # per stage-3/4 N-chunk
        assert K_3 % nb_l3 == 0
        K_3t = K_3 // nb_l3

        def dn(n):
            d = int(n / bdf)
            assert d * b_in_width == n * b_array_width
            return d

        dN_1, dN_2, dN_3, dN_4 = dn(N_1), dn(N_2), dn(N_3), dn(N_4)
        dN_3c, dN_4c = dn(N_3c), dn(N_4c)

        # partition3 runs as one ISGEMM per N-chunk over the resident full-K packed3 (K_3 steps),
        # decoupled from the stage-1-2 tiling. Its B walk reads the chunk's N-run inside each
        # K-block, so the K stride stays the full-N block.
        psum1 = ([M_1 * N_1, K_1], [cd_w // 8, 0])
        psum2 = ([M_2 * N_2, K_2], [cd_w // 8, 0])
        psum3 = ([M_3 * N_3c, K_3], [cd_w // 8, 0])
        psum4 = ([M_4 * N_4c, K_4], [cd_w // 8, 0])

        # cmul2/cmul3 re/im spatial jump: when 2*Lx > 16 the imag rows sit in the second row-block
        # (half the FP8 payload away, like cmul1 at L1=16), not Lx rows within the same block.
        reim_jump2 = L2 * elem_per_bank if 2 * L2 <= 16 else (N_2 // elem_per_bank) * matrix_bytes
        reim_jump3 = L3 * elem_per_bank if 2 * L3 <= 16 else (N_3c // elem_per_bank) * matrix_bytes

        # cmul3 (twiddles3, trailing m4 = L4) — per N-chunk. Two regimes (L4>=slu / L4<slu).
        if L4 >= slu:
            r7_6 = (
                [L4 // slu, L3, A_c, 1],
                [num_banks * matrix_bytes, elem_per_bank, (L4 // elem_per_bank) * matrix_bytes, 0],
                [matrix_bytes, reim_jump3],
            )
            r13_6 = ([L4 // slu, L3, A_c, 1], [2 * slu, 2 * L4, 0, 0])
        else:
            kpg = slu // L4
            r7_6 = (
                [L3 // kpg, A_c, 1, 1],
                [kpg * elem_per_bank, (L4 // elem_per_bank) * matrix_bytes, 0, 0],
                [elem_per_bank, reim_jump3],
            )
            r13_6 = ([L3 // kpg, A_c, 1, 1], [2 * slu, 0, 0, 0])

        streamers = {
            # ===== Stage 1 (per l3-tile): partition 1 =====
            "R11_1": ([2 * L1 * L1p * FP8 // iscore_serial_width], [iscore_serial_width // 8]),
            "R12_1": ([dN_1, M_1, K_1], [b_in_width // 8, 0, dN_1 * b_in_width // 8]),
            "R13_1": psum1,
            "W3_1": psum1,
            # cmul1 (chip layout, trailing m34t = L3t*L4). Un-tiled form with L->Lt, L34->L34t.
            "R7_2": (
                [L2 * L34t * dM // slu, num_banks, L1 // num_banks, 1],
                [num_banks * matrix_bytes, elem_per_bank, num_banks * elem_per_bank, 0],
                [matrix_bytes, Lt * dM * FP8 // 8],
            ),
            "R13_2": ([L2 * L34t * dM // slu, L1, 1, 1], [2 * slu, 2 * L2 * L34t, 0, 0]),
            "W3_2": ([2 * Lt * dM * FP8 // (2 * suc_bc)], [4 * BANK_BYTES]),
            # reorder1 (block-swap deinterleave).
            "R7_2B": ([Lt * dM // slu, 1, 1, 1], [4 * BANK_BYTES, 0, 0, 0], [2 * BANK_BYTES, BANK_BYTES]),
            "W3_2B": ([2 * Lt * dM * FP8 // (2 * suc_bc)], [4 * BANK_BYTES]),
            # ===== Stage 2 (per l3-tile): partition 2 =====
            "R11_3": ([2 * L2 * 2 * L2p * FP8 // iscore_serial_width], [iscore_serial_width // 8]),
            "R12_3": ([dN_2, M_2, K_2], [b_in_width // 8, 0, dN_2 * b_in_width // 8]),
            "R13_3": psum2,
            "W3_3": psum2,
            # cmul2 (twiddles2, trailing m34t). Un-tiled form with L34->L34t, L->Lt.
            "R7_4": (
                [L34t // slu, L2, L1, 1],
                [num_banks * matrix_bytes, elem_per_bank, (L34t // elem_per_bank) * matrix_bytes, 0],
                [matrix_bytes, reim_jump2],
            ),
            "R13_4": ([L34t // slu, L2, L1, 1], [2 * slu, 2 * L34t, 0, 0]),
            "W3_4": ([2 * Lt * dM * FP8 // (2 * suc_bc)], [4 * BANK_BYTES]),
            # reorder2 = scalar m3<->m4 transpose in C (no streamer descriptor).
            # ===== Stage 3 (per N-chunk, K=K_3 over resident packed3): partition 3 =====
            "R11_5": ([2 * L3 * 2 * L3p * FP8 // iscore_serial_width], [iscore_serial_width // 8]),
            "R12_5": ([dN_3c, M_3, K_3], [b_in_width // 8, 0, dN_3 * b_in_width // 8]),
            "R13_5": psum3,
            "W3_5": psum3,
            # cmul3 (twiddles3) — per N-chunk.
            "R7_6": r7_6,
            "R13_6": r13_6,
            "W3_6": ([2 * L * dM * FP8 // (2 * suc_bc * nb_ntile)], [4 * BANK_BYTES]),
            # reorder3 (block-swap deinterleave) — un-tiled (L4 < slu regime only, nb_ntile == 1).
            "R7_6B": ([L * dM // slu, 1, 1, 1], [4 * BANK_BYTES, 0, 0, 0], [2 * BANK_BYTES, BANK_BYTES]),
            "W3_6B": ([2 * L * dM * FP8 // (2 * suc_bc)], [4 * BANK_BYTES]),
            # ===== Stage 4: partition 4 (final, plain) — per N-chunk =====
            "R11_7": ([2 * L4 * 2 * L4p * FP8 // iscore_serial_width], [iscore_serial_width // 8]),
            "R12_7": ([dN_4c, M_4, K_4], [b_in_width // 8, 0, dN_4c * b_in_width // 8]),
            "R13_7": psum4,
            "W3_7": psum4,
        }

        # ---- partition3 K-chunk base offsets (contiguous [re|im] order) ----
        weight3_kchunk_bytes = (2 * L3 * 2 * L3p * FP8 // 8) // nb_l3
        h2_kchunk_bytes = K_3t * dN_3 * b_in_width // 8

        # ---- K-major packed3 layout constants for the scalar transpose ----
        # packed3 byte = k*pk_kstride + (N//bdf)*bw + (N%bdf)*aw + c_within (c = reim*L3 + m3, k = c//slu).
        pk_bw = b_in_width // 8                 # bytes per B-fetch element (= 4*BANK*bdf)
        pk_aw = slu * FP8 // 8                  # bytes per array column = one K-step's contraction
        pk_bdf = pk_bw // pk_aw                 # N's packed per B-fetch (downsize factor)
        pk_kstride = dN_3 * b_in_width // 8     # one K-step's full-N block (= h2_kchunk for K_3t=1)
        # packed4 K-major (partition4 contraction = m4); same formula as packed3 with
        # (reim*L4 + m4), no transpose (m4 already inner). Chunk-local (full at nb_ntile=1).
        pk_kstride4 = dN_4c * b_in_width // 8
        # partition3 B-walk base offset per N-chunk (the chunk's N-run inside each K-block).
        pk3_chunk_off = dN_3c * b_in_width // 8
        # packed2 K-tile stride: at 2*L2 > slu the re/im halves are separate K-tiles that reorder1
        # splits via DMA (the block-swap NOOP only covers 2*L2 <= slu, where one tile holds both).
        pk_kstride2 = dN_2 * b_in_width // 8
        n2_groups = Lt * dM // slu

        # ---- H2 staging (per l3-tile): scatter the transposed packed_tile into the L3 packed3
        # m3/[re|im] regions (one 2-D scatter over N per re/im) ----
        n_full = L1 * L2 * L4 * dM
        pk_re_chunk = L3t * FP8 // 8                 # one N's re (or im) m3-block in packed_tile
        pk_src_stride = 2 * L3t * FP8 // 8           # packed_tile per-N stride (re+im)
        l3_dst_stride = 2 * L3 * FP8 // 8            # L3 packed3 per-N stride (re+im, full K)
        l3_im_off = L3 * FP8 // 8                    # im sub-block within each N's 2*L3 K-run
        full_packed3_bytes = 2 * L * dM * FP8 // 8   # assembled partition3 input (TCDM)

        # ---- buffer sizes ----
        len_weight1 = 2 * L1 * L1p * FP8 // 8
        len_weight2 = 2 * L2 * 2 * L2p * FP8 // 8
        len_weight3 = 2 * L3 * 2 * L3p * FP8 // 8
        len_weight4 = 2 * L4 * 2 * L4p * FP8 // 8
        len_in = L * dModel * FP8 // 8
        len_in_slice = L * dM * FP8 // 8
        len_partition4_out = 2 * L * dModel * BF16 // 8

        slot_size_tile = align64(2 * Lt * dM * BF16 // 8)    # gemm1/2 psum per stage-1-2 tile
        hsize_tile = slot_size_tile // 2                      # FP8 cmul/noop scratch per tile
        packed_tile_bytes = align64(2 * Lt * dM * FP8 // 8)  # scalar-transpose output per tile
        slot_size_full = align64(2 * L * dM * BF16 // 8)     # full partition3/4 psum
        hsize_full = align64(2 * L * dM * FP8 // 8)          # full FP8 cmul/noop scratch
        slot_size_chunk = align64(2 * L * dM * BF16 // (8 * nb_ntile))  # per-chunk partition3/4 psum
        hsize_chunk = align64(2 * L * dM * FP8 // (8 * nb_ntile))       # per-chunk H3 / packed4

        # ---- l3-tile gather of input + twiddles into tile-local buffers (L3 -> L3t) ----
        # dft_in chip layout [d][jc][m2][m1] (jc = m4*L3 + m3); gather the tile's m3-block per (d,m4).
        mu = slu  # m1-within-tile = L1 = seqLenUnroll
        in_gather_chunk = L3t * L2 * mu * FP8 // 8
        in_gather_m4_stride = L3 * L2 * mu * FP8 // 8
        in_gather_d_stride = L34 * L2 * mu * FP8 // 8
        in_gather_reps_m4 = L4
        in_gather_reps_d = dM
        in_tile_bytes = Lt * dM * FP8 // 8
        # twiddles1 [k1][jc][m2], 2 bytes/entry (re,im interleaved): tile m3-block per (k1, m4).
        # Under m4-tiling the tile's jc-slice is contiguous, so the C side gathers one chunk of
        # tw1_tile_bytes/L1 per k1 (and tw2_tile_bytes/L2 per k2) instead of the m3-block walk.
        tw_e = 2 * FP8 // 8
        tw1_gather_chunk = L3t * L2 * tw_e
        tw1_gather_m4_stride = L3 * L2 * tw_e
        tw1_k1_stride = L34 * L2 * tw_e
        tw1_tile_bytes = L1 * L2 * L34t * tw_e   # Lt * tw_e
        # twiddles2 [k2][jc], 2 bytes/entry: tile m3-block per (k2, m4).
        tw2_gather_chunk = L3t * tw_e
        tw2_gather_m4_stride = L3 * tw_e
        tw2_k2_stride = L34 * tw_e
        tw2_tile_bytes = L2 * L34t * tw_e

        # ---- output scatter: partition4_out (final). FP8 real data in first half; M_4 row-blocks,
        # d outer column factor; slice s owns dM contiguous channels inside each full row-block. ----
        out_block_full = dModel * L1 * L2 * L3 * FP8 // 8
        out_block_slice = dM * L1 * L2 * L3 * FP8 // 8
        out_nblk = M_4
        # P4 N-chunk scatter to L3: chunk c owns p4c_blk_bytes inside each of the M_4 row-blocks.
        # Blocks are sized on the FP8 payload (first half of the psum buffer), not the BF16 span.
        p4_blk_bytes = 2 * L * dM * FP8 // (8 * M_4)
        p4c_blk_bytes = p4_blk_bytes // nb_ntile

        specs = [
            ("weight1", len_weight1), ("weight2", len_weight2),
            ("weight3", len_weight3), ("weight4", len_weight4),
            ("in", len_in),
            ("twiddles1", 2 * L * FP8 // 8),
            ("twiddles2", 2 * L2 * L34 * FP8 // 8),
            ("twiddles3", 2 * L3 * L4 * FP8 // 8),
            ("partition4_out", len_partition4_out),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)

        slice_scalars = {
            "nb_d": nb_d, "dModel_slice": dM, "nb_l3": nb_l3, "l3_tile": l3_tile, "nb_ntile": nb_ntile,
            "nb_m4": nb_m4, "L4t": L4t, "nb_12": nb_12,
            "length_in_slice": len_in_slice,
            "slot_size_tile": slot_size_tile, "hsize_tile": hsize_tile,
            "packed_tile_bytes": packed_tile_bytes, "in_tile_bytes": in_tile_bytes,
            "tw1_tile_bytes": tw1_tile_bytes, "tw2_tile_bytes": tw2_tile_bytes,
            "slot_size_full": slot_size_full, "hsize_full": hsize_full,
            "slot_size_chunk": slot_size_chunk, "hsize_chunk": hsize_chunk,
            "gemm_N1": N_1, "gemm_N2": N_2, "gemm_N3": N_3c, "gemm_N4": N_4c,
            "full_packed3_bytes": full_packed3_bytes,
            # gather strides
            "in_gather_chunk": in_gather_chunk, "in_gather_m4_stride": in_gather_m4_stride,
            "in_gather_d_stride": in_gather_d_stride, "in_gather_reps_m4": in_gather_reps_m4,
            "in_gather_reps_d": in_gather_reps_d,
            "tw1_gather_chunk": tw1_gather_chunk, "tw1_gather_m4_stride": tw1_gather_m4_stride,
            "tw1_k1_stride": tw1_k1_stride,
            "tw2_gather_chunk": tw2_gather_chunk, "tw2_gather_m4_stride": tw2_gather_m4_stride,
            "tw2_k2_stride": tw2_k2_stride,
            # scalar transpose + staging
            "L34t": L34t, "L3t": L3t,
            "pk_re_chunk": pk_re_chunk, "pk_src_stride": pk_src_stride,
            "l3_dst_stride": l3_dst_stride, "l3_im_off": l3_im_off, "n_full": n_full,
            # partition3 K-chunk offsets + K-major packed3 layout
            "weight3_kchunk_bytes": weight3_kchunk_bytes, "h2_kchunk_bytes": h2_kchunk_bytes,
            "pk_bw": pk_bw, "pk_aw": pk_aw, "pk_bdf": pk_bdf, "pk_kstride": pk_kstride,
            "pk_kstride4": pk_kstride4, "n4_full": dM * L1 * L2 * L3, "n4_chunk": N_4c,
            "pk3_chunk_off": pk3_chunk_off,
            "pk_kstride2": pk_kstride2, "n2_groups": n2_groups,
            "p4_blk_bytes": p4_blk_bytes, "p4c_blk_bytes": p4c_blk_bytes,
            # output scatter
            "out_block_full": out_block_full, "out_block_slice": out_block_slice, "out_nblk": out_nblk,
        }
        scalars = {**lengths, **deltas, **slice_scalars}

        test_data = {
            **{
                name: "uint8_t"
                for name in (
                    "dft_weight1", "dft_weight2", "dft_weight3", "dft_weight4", "dft_in",
                    "twiddles1", "twiddles2", "twiddles3",
                    "partition1_expected", "hadamard1_expected", "partition2_expected",
                    "hadamard2_expected", "partition3_expected", "hadamard3_expected",
                    "partition4_expected",
                )
            }
        }
        tests = {"expected": 2 * L * dModel}
        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
