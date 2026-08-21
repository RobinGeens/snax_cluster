#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Parallel OSGEMM + ISGEMM, tiled along dInner, with the AGU XOR bank swizzle on the
# live streamer ports. See docs/dataflow/22_agu_xor_swizzle.md.

import pathlib
import random
import sys
import os
import hjson

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
# Path in Occamy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import DataGeneratorBase, NB_TEST_SAMPLES, FP8, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]

SB = 256  # swizzle permutes 32 B chunks within each 256 B TCDM row
WIN = 2048  # swizzle key period: bits [10:8] of the address


def swz(a: int) -> int:
    """XOR bank swizzle, must match the AGU's XorBankSwizzleMapping: addr[7:5] ^= addr[10:8]."""
    return (a & ~0xE0) | ((((a >> 5) ^ (a >> 8)) & 0x7) << 5)


class DataGenerator(DataGeneratorBase):
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
        self.compute_layout()
        self.build_osgemm_data()
        self.build_isgemm_data()
        self.emit_layout()

    # ---------------- TCDM layout with per-stream swizzle phases ----------------

    def _alloc(self, size: int, phase: int) -> int:
        assert size % SB == 0, f"buffer size {size} is not a multiple of {SB}"
        off = self._cursor + ((phase * SB - self._cursor) % WIN)
        self._cursor = off + size
        return off

    def compute_layout(self):
        L = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        dInner = self.kwargs["dInner"]
        nb = self.kwargs["nb_tiles"]

        self.len_a_os = L * dModel * FP8 // 8
        self.len_b_os_tile = dModel * dInner * FP8 // 8 // nb
        self.len_d_os_tile = L * dInner * FP8 // 8 // nb
        self.len_a_is_tile = L * dInner * FP8 // 8 // nb
        self.len_b_is_tile = dInner * dModel * FP8 // 8 // nb
        self.len_cd = L * dModel * BF16 // 8

        # One distinct bits[10:8] phase per concurrently-live stream, so lockstep sweeps
        # land in disjoint bank groups. Ping-pong halves share their stream's phase. The
        # psum pair R13/W3 stays identity-mapped (a swizzled same-buffer RMW pair
        # collides with itself, see the docs).
        self._cursor = 0
        self.off_a_os = self._alloc(self.len_a_os, 0)  # R0
        self.off_b_os = [self._alloc(self.len_b_os_tile, 1) for _ in range(2)]  # R1
        self.off_a_is = [self._alloc(self.len_a_is_tile, 2) for _ in range(2)]  # R11
        self.off_b_is = [self._alloc(self.len_b_is_tile, 3) for _ in range(2)]  # R12
        self.off_cd = self._alloc(self.len_cd, 4)  # R13 + W3, identity
        self.off_d_os = [self._alloc(self.len_d_os_tile, 5) for _ in range(2)]  # W0
        self.peak_bytes = self._cursor + WIN  # + slack for aligning the runtime base

    def emit_layout(self):
        self.format("uint32_t", "SWZ_off_a_os", self.off_a_os)
        for i in range(2):
            self.format("uint32_t", f"SWZ_off_b_os_{i}", self.off_b_os[i])
            self.format("uint32_t", f"SWZ_off_a_is_{i}", self.off_a_is[i])
            self.format("uint32_t", f"SWZ_off_b_is_{i}", self.off_b_is[i])
            self.format("uint32_t", f"SWZ_off_d_os_{i}", self.off_d_os[i])
        self.format("uint32_t", "SWZ_off_cd", self.off_cd)
        self.lines_params.append(f"#define L1_TCDM_PEAK_BYTES {self.peak_bytes}u")

    # ---------------- pre-swizzled DMA images and sample indices ----------------

    @staticmethod
    def _swizzle_block(raw: bytes, base: int) -> bytes:
        assert base % SB == 0, f"buffer base {base} is not {SB} B aligned"
        assert len(raw) % SB == 0
        out = bytearray(len(raw))
        for o in range(0, len(raw), 32):
            p = swz(base + o) - base
            out[p : p + 32] = raw[o : o + 32]
        return bytes(out)

    def emit_swizzled(self, ctype: str, name: str, elem_bytes: int, tile_bases: list[int], tile_len: int | None = None):
        """Emit tensor `name` with each tile pre-swizzled for its TCDM destination, so a
        plain 1-D DMA lands every element at its swizzled physical address.
        tile_len in bytes; tile i is DMA'd to tile_bases[i % len(tile_bases)]."""
        vals = self._read_data_int(f"{name}.bin")
        raw = b"".join(int(v).to_bytes(elem_bytes, "little") for v in vals)
        tl = tile_len if tile_len is not None else len(raw)
        assert len(raw) % tl == 0
        out = b"".join(
            self._swizzle_block(raw[i * tl : (i + 1) * tl], tile_bases[i % len(tile_bases)])
            for i in range(len(raw) // tl)
        )
        elems = [int.from_bytes(out[i * elem_bytes : (i + 1) * elem_bytes], "little") for i in range(len(vals))]
        self.format_vector(ctype, name, elems)

    def emit_samples_swz(self, mode_id: int, tensor: str, size: int, tile_bases: list[int], tile_len: int | None = None):
        """Byte-sampled result check: logical index (into the golden) plus its swizzled
        twin (into the result image, which lives at the swizzled addresses)."""
        idx = sorted(random.randint(0, size - 1) for _ in range(NB_TEST_SAMPLES))
        tl = tile_len if tile_len is not None else size
        idx_swz = []
        for k in idx:
            t, r = divmod(k, tl)
            base = tile_bases[t % len(tile_bases)]
            idx_swz.append(t * tl + swz(base + r) - base)
        self.format_vector("int32_t", f"M{mode_id}_test_samples_{tensor}", idx)
        self.format_vector("int32_t", f"M{mode_id}_test_samples_{tensor}_swz", idx_swz)

    # ---------------- modes ----------------

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

        streamers = {
            "R0": (
                [K, M, N_tile],
                [a_in_width // 8, K * a_in_width // 8, 0],
            ),
            "R1": (
                [downsized_K, M, N_tile],
                [b_in_width // 8, 0, downsized_K * b_in_width // 8],
            ),
            "W0": (
                [(d_array_width // oscore_serial_width) * M * N_tile],
                [oscore_serial_width // 8],
            ),
        }

        len_a = M * K * a_in_width // 8
        len_b = K * N * b_array_width // 8
        len_d = M * N * d_array_width // 8
        assert (len_a, len_b // nb_tiles, len_d // nb_tiles) == (
            self.len_a_os,
            self.len_b_os_tile,
            self.len_d_os_tile,
        )

        specs = [("a", len_a), ("b", len_b), ("d", len_d)]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        tile_scalars = {"length_b_tile": len_b // nb_tiles, "length_d_tile": len_d // nb_tiles}
        scalars = {**lengths, **deltas, **tile_scalars}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data={}, tests={})

        self.emit_swizzled("uint8_t", "M3_A", 1, [self.off_a_os])
        self.emit_swizzled("uint8_t", "M3_B", 1, self.off_b_os, self.len_b_os_tile)
        self.read_and_format_vector(mode_id, "uint8_t", "D")
        self.emit_samples_swz(mode_id, "D", len_d, self.off_d_os, self.len_d_os_tile)

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
            "R11": (
                [K_t * (seqLen // seqLenUnroll) * (a_in_width // iscore_serial_width)],
                [iscore_serial_width // 8],
            ),
            "R12": (
                [downsized_dModel, seqLen // seqLenUnroll, K_t],
                [b_in_width // 8, 0, downsized_dModel * b_in_width // 8],
            ),
            "R13": psum_bounds_and_strides,
            "W3": psum_bounds_and_strides,
        }

        len_a = seqLen * dInner * FP8 // 8
        len_b = dInner * dModel * FP8 // 8
        len_cd = seqLen * dModel * BF16 // 8
        assert (len_a // nb_tiles, len_b // nb_tiles, len_cd) == (
            self.len_a_is_tile,
            self.len_b_is_tile,
            self.len_cd,
        )

        specs = [("a", len_a), ("b", len_b), ("cd", len_cd)]
        lengths, deltas = self._collect_lengths_and_deltas(specs)
        tile_scalars = {"length_a_tile": len_a // nb_tiles, "length_b_tile": len_b // nb_tiles}
        scalars = {**lengths, **deltas, **tile_scalars}

        self.build_mode(mode_id, streamers, scalars=scalars, test_data={}, tests={})

        self.emit_swizzled("uint8_t", "M4_A", 1, self.off_a_is, self.len_a_is_tile)
        self.emit_swizzled("uint8_t", "M4_B", 1, self.off_b_is, self.len_b_is_tile)
        # CD (psum) is identity-mapped: plain images and plain sample indices
        self.read_and_format_vector(mode_id, "uint16_t", "C")
        self.read_and_format_vector(mode_id, "uint8_t", "D")
        idx = sorted(random.randint(0, len_cd - 1) for _ in range(NB_TEST_SAMPLES))
        self.format_vector("int32_t", f"M{mode_id}_test_samples_D", idx)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
