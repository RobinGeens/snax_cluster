#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Tiled, double-buffered, pipelined version of the Mamba main program. Both
# Phase 1 and Phase 2 are tiled along the dInner dimension. Each per-phase
# pipeline first finishes all tiles of Phase 1, then all tiles of Phase 2.
# Tiling strategy:
#   Phase 1
#     - in proj x        : tiled in dInner (osCore output)
#     - x_proj           : tiled in dInner (IS GeMM K-dim, accumulates psum)
#     - x_proj outputs   : NOT tiled (delta, B, C all stay in iscore_out)
#   Phase 2
#     - in proj z        : tiled in dInner (osCore output)
#     - delta proj       : tiled in dInner (switchCore output)
#     - SUC              : tiled in dInner (consumes/produces dInner-sized x,y,z)
#     - out proj         : tiled in dInner (IS GeMM K-dim, accumulates psum)
#     - isCore output    : NOT tiled (kept in TCDM, accumulates across tiles)

import pathlib
import random
import sys
import os
import math
import hjson

# Fix the Python RNG so test_samples_* arrays are stable across regenerations.
# (chisel-ssm's scala.util.Random still varies the .bin contents between sbt runs,
# but those don't depend on this Python script's seed.)
random.seed(0)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
# Path in Occamy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from datagen_base import BANKWIDTH, DataGeneratorBase, BANK_BYTES, FP8, BF16  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]


class DataGenerator(DataGeneratorBase):
    APP_NAME = "main-tiled"

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        self.phase1_scalars: dict[str, int] = {}
        self.phase2_scalars: dict[str, int] = {}
        # Not all parameters are propagated to scala, so read them from the local params hjson file
        local_params_path = pathlib.Path(__file__).resolve().parent / "params_in.hjson"
        with local_params_path.open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.save_params()
        self.check_tiling_constraints()
        self.build_Phase1_data()
        self.build_Phase2_data()
        self.emit_l1_usage_comment()

    # snax_simbacore_cluster.hjson: tcdm.size = 512 kB.
    TCDM_BYTES = 512 * 1024

    def emit_l1_usage_comment(self):
        """Emit expected peak L1 footprint and hard-assert it fits in TCDM.

        Layout assumed by main.c after the x-tiling refactor:
          - oscore_in, iscore_out_P1, z, y, iscore_out_P2 stay FULL.
          - conv_out (= P2 x) is tiled: P1 writes a tile-sized slot then
            DMAs to L3; P2 DMAs a tile-sized slot back from L3. Two
            ping-pong slots in the shared scratch region (sized length_conv_out_tile each).
        """
        def align64(value: int) -> int:
            return (value + 63) & ~63

        def pingpong_bytes(tile_lengths: list[int]) -> int:
            """Bytes consumed by two aligned ping-pong slots per tiled tensor."""
            cursor = 0
            for length in tile_lengths:
                # First buffer
                cursor = align64(cursor + length)
                # Second buffer
                cursor = align64(cursor + length)
            return cursor

        # Shared region in test_phase1_and_2 (always live across both phases).
        # conv_out is no longer FULL: it lives in the ping-pong region as a tile-sized
        # slot pair (P1 writes, DMAs to L3; P2 DMAs back into the same slots).
        # iscore_out_P1 now has TWO buffers: psum (R13/W3 non-final) + final (W3 final
        # = bank-transposed output that P2's R2/R7 reads). This split is needed because the
        # final-tile TRANSPOSE W3 would otherwise overwrite psum_partial(n,m) before
        # IS-core processes element (n,m). See main.c for the explanation.
        # z and y are also no longer FULL: they live in the P2 ping-pong region as
        # tile-sized slot pairs (W0/R10 and W2/R11 access the SAME slot within a kernel,
        # then DM spills the slot to L3 once the kernel is done).
        # iscore_out_P1_psum and iscore_out_P2 OVERLAP: psum is dead after P1's final
        # tile (R13/W3 redirect to iscore_out_P1_final on that tile); iscore_out_P2 is
        # only used in P2 (bias preload overwrites psum's stale data). Reserve max(.,.)
        # bytes for the shared block.
        iscore_p1_p2_shared = max(
            self.phase1_scalars["length_iscore_out"],   # P1 psum_buf
            self.phase2_scalars["length_iscore_out"],   # P2 iscore_out
        )
        shared_bytes = (
            self.phase1_scalars["length_oscore_in"]
            + iscore_p1_p2_shared                              # P1 psum / P2 iscore_out (overlap)
            + self.phase1_scalars["length_iscore_out"]         # P1 final (separate; transposed buffer)
        )

        # Ping-pong scratch used only by Phase 1. conv_out_tile is in the ping-pong area
        # so it can be double-buffered against the L3 DMA-out.
        p1_pingpong_bytes = pingpong_bytes(
            [
                self.phase1_scalars["length_oscore_weight_tile"],
                self.phase1_scalars["length_conv_weight_tile"],
                self.phase1_scalars["length_conv_bias_tile"],
                self.phase1_scalars["length_iscore_weight_tile"],
                self.phase1_scalars["length_conv_out_tile"],  # P1 W1 ping-pong, DMA-out to L3
            ]
        )

        # Ping-pong scratch used only by Phase 2 (reuses Phase 1 scratch base).
        # x_tile is the P2-side reuse of conv_out_tile slots (DMA-in from L3 per K-tile).
        # z_tile and y_tile are also ping-ponged (W0/R10 + W2/R11 within a kernel, DMA-out
        # to L3 between kernels).
        p2_pingpong_bytes = pingpong_bytes(
            [
                self.phase2_scalars["length_oscore_weight_tile"],
                self.phase2_scalars["length_dt_weight_1_tile"],
                self.phase2_scalars["length_dt_weight_2_tile"],
                self.phase2_scalars["length_dt_bias_tile"],
                self.phase2_scalars["length_A_tile"],
                self.phase2_scalars["length_D_tile"],
                self.phase2_scalars["length_iscore_weight_tile"],
                self.phase1_scalars["length_conv_out_tile"],  # x_tile ping-pong, DMA-in from L3
                self.phase2_scalars["length_z_tile"],         # z_tile ping-pong, DMA-out to L3
                self.phase2_scalars["length_y_tile"],         # y_tile ping-pong, DMA-out to L3
            ]
        )

        # main.c aligns ping-pong base to 64 bytes after the shared region.
        total_l1_bytes = align64(shared_bytes) + max(p1_pingpong_bytes, p2_pingpong_bytes)
        total_l1_kib = total_l1_bytes / 1024
        tcdm_kib = self.TCDM_BYTES / 1024
        print(
            f"// Expected total L1 usage (tiled test_phase1_and_2 layout): {total_l1_bytes} B ({total_l1_kib:.2f} KiB)"
        )
        print(
            f"//   shared FULL (oscore_in + iscore_out_P1 + z + y + iscore_out_P2) = {shared_bytes} B"
        )
        print(
            f"//   pingpong scratch (max of P1/P2 incl. conv_out_tile / x_tile)    = {max(p1_pingpong_bytes, p2_pingpong_bytes)} B"
        )
        # Hard-assert fit so silent OOB writes can never reach the simulator.
        # When you see this fire, lower seqLen or raise nb_tiles, or extend the tiling
        # strategy (iscore_out, z, y currently stay FULL — see params_in.hjson notes).
        assert total_l1_bytes <= self.TCDM_BYTES, (
            f"main-tiled L1 footprint {total_l1_bytes} B ({total_l1_kib:.2f} KiB) "
            f"exceeds TCDM budget {self.TCDM_BYTES} B ({tcdm_kib:.0f} KiB). "
            f"shared FULL = {shared_bytes} B, pingpong = {max(p1_pingpong_bytes, p2_pingpong_bytes)} B. "
            f"Lower seqLen, raise nb_tiles, or extend tiling (z/y/iscore_out_P2 stay FULL today)."
        )

    def save_params(self):
        """Saves params to self as shorthand notation"""
        self.seqLen = self.kwargs["seqLen"]
        self.dModel = self.kwargs["dModel"]
        self.dInner = self.kwargs["dInner"]
        self.dtRank = self.kwargs["dtRank"]
        self.dConv = self.kwargs["dConv"]
        self.dState = self.kwargs["dState"]
        self.xProjDim = self.kwargs["xProjDim"]
        self.seqLenUnroll = self.kwargs["seqLenUnroll"]
        self.dInnerUnroll = self.kwargs["dInnerUnroll"]
        self.dtRankUnroll = self.kwargs["dtRankUnroll"]
        self.convUnroll = self.kwargs["convUnroll"]
        self.delaySU = self.kwargs["delaySU"]
        self.oscore_serial_width = self.kwargs["oscore_serial_width"]
        self.switchcore_width = self.kwargs["switchcore_width"]
        self.iscore_serial_width = self.kwargs["iscore_serial_width"]
        self.suc_serial_width_A = self.kwargs["suc_serial_width_A"]
        self.suc_serial_width_BC = self.kwargs["suc_serial_width_BC"]  # Streamer width is 2x this value!
        self.gemm_weight_width = self.kwargs["gemm_weight_width"]
        self.weight_downsize_factor = self.gemm_weight_width / (self.dInnerUnroll * FP8)  # >= 1
        self.downsized_dModel = int(self.dModel / self.weight_downsize_factor)
        self.downsized_xProjDim = int(self.xProjDim / self.weight_downsize_factor)
        assert self.downsized_dModel * self.gemm_weight_width == self.dModel * (self.dInnerUnroll * FP8)
        assert self.downsized_xProjDim * self.gemm_weight_width == self.xProjDim * (self.dInnerUnroll * FP8)

        # Tiling
        self.nb_tiles = self.kwargs["nb_tiles"]
        self.dInner_tile = self.dInner // self.nb_tiles

    def check_tiling_constraints(self):
        """Verify all dInner-derived dimensions remain integer (and HW-legal) after tiling."""
        nb = self.nb_tiles
        # Per-tile dInner must contain a whole number of dInnerUnroll lanes
        assert self.dInner % (self.dInnerUnroll * nb) == 0, (
            f"dInner ({self.dInner}) must be a multiple of dInnerUnroll * nb_tiles "
            f"({self.dInnerUnroll} * {nb} = {self.dInnerUnroll * nb})"
        )
        # Conv weight bound: dConv * dInner * FP8 // BANKWIDTH must split cleanly
        assert (self.dConv * self.dInner * FP8) % (BANKWIDTH * nb) == 0
        # SwitchCore bias / SUC D bound: dInner * FP8 // BANKWIDTH must split
        assert (self.dInner * FP8) % (self.switchcore_width * nb) == 0
        # SUC BC outermost bound (irrelevant axis): dInner // delaySU
        assert self.dInner % (self.delaySU * nb) == 0
        # SwitchCore weights bound: dInner / convUnroll factor
        assert self.dInner % (self.convUnroll * nb) == 0

    def get_safe_to_start_delay(self, dInner: int):
        """Per-tile (or per-phase) safe-to-start delay for R10 and R11. Same algorithm
        as in main/data/datagen.py but parameterised on the (possibly tiled) dInner so
        the counters scale with the per-kernel workload.
        """
        MARGIN = 0.2  # 20%
        gemm_cycles_per_tile = self.dModel
        gemm_total_nb_tiles = (self.seqLen // self.seqLenUnroll) * (dInner // self.dInnerUnroll)
        gemm_cycles = gemm_total_nb_tiles * gemm_cycles_per_tile

        suc_cycles_per_element = 1
        suc_total_nb_elements = self.seqLen * dInner
        suc_cycles = suc_total_nb_elements * suc_cycles_per_element

        gemm_window_cnt = self.seqLen // self.seqLenUnroll  # OS core tiles
        suc_window_cnt = self.seqLen * self.dInnerUnroll  # SUC output elements

        suc_delta = (gemm_cycles - suc_cycles) / gemm_cycles_per_tile  # [tiles]
        iscore_delta = (suc_cycles - gemm_cycles) / suc_cycles_per_element  # [elements]

        suc_safe_to_start = math.ceil(max(gemm_window_cnt, suc_delta) * (1 + MARGIN))
        iscore_safe_to_start = math.ceil(max(suc_window_cnt, iscore_delta) * (1 + MARGIN))

        print(f"// DEBUG safe-to-start delays (per-tile, dInner={dInner}):")
        print(f"//      OScore cycles: {gemm_cycles}")
        print(f"//      OScore window: {gemm_window_cnt}")
        print(f"//      SUC cycles: {suc_cycles}")
        print(f"//      SUC delta: {suc_delta}")
        print(f"//      SUC safe to start: {suc_safe_to_start}")
        print(f"//      SUC window: {suc_window_cnt}")
        print(f"//      IScore delta: {iscore_delta}")
        print(f"//      IScore safe to start: {iscore_safe_to_start}")

        return int(min(suc_safe_to_start, gemm_total_nb_tiles)), int(min(iscore_safe_to_start, suc_total_nb_elements))

    # =========================================================================
    # Phase 1
    # =========================================================================
    def build_Phase1_data(self):
        mode_id = 1
        assert f"M{mode_id}_PHASE1" in self.kwargs, "verify mode_id"
        assert self.switchcore_width == BANKWIDTH

        # All bounds touching the dInner axis are scaled to the per-tile workload.
        # Strides are unchanged (within a tile the access pattern is identical).
        dInner_t = self.dInner_tile
        N_t = dInner_t // self.dInnerUnroll  # per-tile osCore N tiles (equiv. K-tiles for IS GeMM)

        streamers = {
            "R0": (  # osCore in (FULL input, base ptr unchanged across tiles)
                [
                    self.dModel,  # K
                    self.seqLen // self.seqLenUnroll,  # M
                    N_t,  # N (per-tile)
                ],
                [
                    self.seqLenUnroll * FP8 // 8,
                    self.dModel * self.seqLenUnroll * FP8 // 8,
                    0,
                ],
            ),
            "R1": (  # osCore weight (per-tile slice along N=dInner)
                [
                    self.downsized_dModel,  # K
                    self.seqLen // self.seqLenUnroll,  # M
                    N_t,  # N (per-tile)
                ],
                [
                    self.gemm_weight_width // 8,
                    0,
                    self.downsized_dModel * self.gemm_weight_width // 8,
                ],
            ),
            "R3": (  # conv (switchCore) weight: layout is row-major [dInner, dConv]
                [self.dConv * dInner_t * FP8 // BANKWIDTH],
                [BANK_BYTES],
            ),
            "R4": (  # conv (switchCore) bias: layout is row-major [dInner]
                [dInner_t * FP8 // BANKWIDTH],
                [BANK_BYTES],
            ),
            "R12": (  # iscore weight (x_proj, K-tiled along dInner, accumulated by R13/W3)
                [
                    self.downsized_xProjDim,  # N
                    self.seqLen // self.seqLenUnroll,  # M
                    N_t,  # K (per-tile)
                ],
                [
                    self.gemm_weight_width // 8,
                    0,
                    self.downsized_xProjDim * self.gemm_weight_width // 8,
                ],
            ),
            "R13": (  # isCore psum (FULL output, accumulates across tiles in place)
                [
                    (self.seqLen // self.seqLenUnroll) * self.xProjDim,
                    N_t,  # K (per-tile)
                ],
                [
                    self.seqLenUnroll * BF16 // 8,
                    0,
                ],
            ),
            "W1": (  # conv output (per-tile slice along dInner)
                [self.seqLen * dInner_t * FP8 // BANKWIDTH],
                [BANK_BYTES],
            ),
            "W3": (  # isCore output: same as psum reader R13 (FULL, in-place accumulation)
                [
                    (self.seqLen // self.seqLenUnroll) * self.xProjDim,
                    N_t,
                ],
                [
                    self.seqLenUnroll * BF16 // 8,
                    0,
                ],
            ),
        }

        # ---------- Buffer sizes -------------------------------------------------
        # Full sizes are kept (used for L3 buffers / DMA copies) and we additionally
        # emit per-tile sizes so main.c can issue per-iteration DMAs without
        # arithmetic.
        len_oscore_in = self.seqLen * self.dModel * FP8 // 8  # not tiled
        len_oscore_weight = self.dModel * self.dInner * FP8 // 8
        len_conv_weight = self.dInner * self.dConv * FP8 // 8
        len_conv_bias = self.dInner * FP8 // 8
        len_conv_out = self.seqLen * self.dInner * FP8 // 8
        len_iscore_weight = self.dInner * self.xProjDim * FP8 // 8
        len_iscore_out = self.seqLen * self.xProjDim * BF16 // 8  # not tiled

        nb = self.nb_tiles
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
            "dInner_tile": dInner_t,
            "length_oscore_weight_tile": len_oscore_weight // nb,
            "length_conv_weight_tile": len_conv_weight // nb,
            "length_conv_bias_tile": len_conv_bias // nb,
            "length_conv_out_tile": len_conv_out // nb,
            "length_iscore_weight_tile": len_iscore_weight // nb,
        }
        scalars = {**lengths, **deltas, **tile_scalars}
        self.phase1_scalars = scalars.copy()

        tests = {"conv_out": self.seqLen * self.dInner, "iscore_out": self.seqLen * self.xProjDim}

        test_data = {
            name: "uint8_t"
            for name in (
                "oscore_in",
                "oscore_weight",
                "conv_weight",
                "conv_bias",
                "conv_out",
                "iscore_weight",
                "iscore_out",
            )
        }

        test_data["iscore_bias"] = "uint16_t"

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)

    # =========================================================================
    # Phase 2
    # =========================================================================
    def build_Phase2_data(self):
        mode_id = 2
        assert f"M{mode_id}_PHASE2" in self.kwargs, "verify mode_id"
        assert self.dtRank * FP8 % self.switchcore_width == 0, "dtRank must be divisible by switchCore elem/cc in"
        assert self.switchcore_width == BANKWIDTH, "switchcore_width must match bank width"

        dInner_t = self.dInner_tile
        N_t = dInner_t // self.dInnerUnroll

        suc_parallel_widthA = self.dState * FP8
        switchcore_parallel_width_W1 = self.convUnroll * self.dConv * FP8
        switchcore_parallel_width_W2 = self.convUnroll * (self.dtRankUnroll - self.dConv) * FP8
        oscore_parallel_width_d = self.seqLenUnroll * self.dInnerUnroll * FP8
        iscore_parallel_width_d = self.seqLenUnroll * self.dInnerUnroll * FP8

        # Reads out a layout that is stored in convFormat, in SUC format ordering. Per-tile we only iterate the
        # outer dimension (windows per tensor) over dInner_tile, leaving the other three intact.
        bounds_conv_to_suc = [
            (self.convUnroll * self.seqLenUnroll) // (BANKWIDTH // FP8),
            self.seqLen // self.seqLenUnroll,
            self.dInnerUnroll // self.convUnroll,
            dInner_t // self.dInnerUnroll,  # windows per tensor (per-tile)
        ]
        strides_conv_to_suc = [
            BANK_BYTES,
            self.seqLenUnroll * self.dInnerUnroll * FP8 // 8,
            self.convUnroll * self.seqLenUnroll * FP8 // 8,
            self.seqLen * self.dInnerUnroll * FP8 // 8,
        ]

        streamers = {
            "R0": (  # osCore in (FULL input, base ptr unchanged)
                [self.dModel, self.seqLen // self.seqLenUnroll, N_t],
                [self.seqLenUnroll * FP8 // 8, self.dModel * self.seqLenUnroll * FP8 // 8, 0],
            ),
            "R1": (  # osCore weight (z proj, tiled along N=dInner)
                [self.downsized_dModel, self.seqLen // self.seqLenUnroll, N_t],
                [self.gemm_weight_width // 8, 0, self.downsized_dModel * self.gemm_weight_width // 8],
            ),
            "R2": (  # switchCore in (FULL dt slice from iscore_out_P1, base ptr unchanged)
                [
                    self.dtRank * FP8 // self.switchcore_width,  # K
                    self.seqLenUnroll,
                    self.seqLen // self.seqLenUnroll,
                    dInner_t // self.convUnroll,  # N (irrelevant dim, per-tile)
                ],
                [
                    (self.switchcore_width // 8) * self.seqLenUnroll,
                    BANK_BYTES,
                    self.seqLenUnroll * self.xProjDim * FP8 // 8,
                    0,
                ],
                self.seqLenUnroll * BANK_BYTES,
            ),
            "R3": (  # switchCore weight (partition 1)
                [
                    (self.dtRank // self.dtRankUnroll)
                    * (dInner_t // self.convUnroll)
                    * (switchcore_parallel_width_W1 // self.switchcore_width)
                ],
                [self.switchcore_width // 8],
            ),
            "R4": (  # switchCore bias
                [(dInner_t * FP8) // self.switchcore_width],
                [self.switchcore_width // 8],
            ),
            "R5": (  # switchCore weight (partition 2)
                [
                    (self.dtRank // self.dtRankUnroll)
                    * (dInner_t // self.convUnroll)
                    * (switchcore_parallel_width_W2 // self.switchcore_width)
                ],
                [self.switchcore_width // 8],
            ),
            "R6": (  # SUC A
                [dInner_t * (suc_parallel_widthA // self.suc_serial_width_A)],
                [self.suc_serial_width_A // 8],
            ),
            "R7": (  # SUC BC (FULL, base ptr unchanged; only the irrelevant dInner-axis bound shrinks)
                [
                    (2 * self.dState * FP8) // (2 * self.suc_serial_width_BC),
                    self.seqLenUnroll,
                    self.seqLen // self.seqLenUnroll,
                    dInner_t // self.delaySU,  # irrelevant dimension, per-tile
                ],
                [
                    (2 * self.suc_serial_width_BC // 8) * self.seqLenUnroll,
                    BANK_BYTES,
                    self.seqLenUnroll * self.xProjDim * FP8 // 8,
                    0,
                ],
                self.seqLenUnroll * BANK_BYTES,
            ),
            "R8": (  # SUC D
                [dInner_t * FP8 // BANKWIDTH],
                [BANK_BYTES],
            ),
            "R9": (bounds_conv_to_suc, strides_conv_to_suc),  # SUC x
            "R10": (bounds_conv_to_suc, strides_conv_to_suc),  # SUC z
            "R11": (  # iscore in. Stored in convFormat
                [
                    (dInner_t // self.dInnerUnroll)
                    * (self.seqLen // self.seqLenUnroll)
                    * (iscore_parallel_width_d // self.iscore_serial_width)
                ],
                [self.iscore_serial_width // 8],
            ),
            "R12": (  # iscore weight (out_proj, K-tiled along dInner, accumulated by R13/W3)
                [self.downsized_dModel, self.seqLen // self.seqLenUnroll, N_t],
                [self.gemm_weight_width // 8, 0, self.downsized_dModel * self.gemm_weight_width // 8],
            ),
            "R13": (  # isCore psum (FULL, in-place accumulation across tiles)
                [(self.seqLen // self.seqLenUnroll) * self.dModel, N_t],
                [self.seqLenUnroll * BF16 // 8, 0],
            ),
            "W0": (  # osCore out (z, per-tile)
                [
                    (oscore_parallel_width_d // self.oscore_serial_width)
                    * (self.seqLen // self.seqLenUnroll)
                    * (dInner_t // self.dInnerUnroll)
                ],
                [self.oscore_serial_width // 8],
            ),
            "W2": (  # SUC y output (per-tile, written to convFormat)
                bounds_conv_to_suc,
                strides_conv_to_suc,
            ),
            "W3": (  # isCore output: SAME addresses as R13 (FULL, in-place accumulation)
                [(self.seqLen // self.seqLenUnroll) * self.dModel, N_t],
                [self.seqLenUnroll * BF16 // 8, 0],
            ),
        }

        # ---------- Buffer sizes -------------------------------------------------
        len_oscore_in = self.seqLen * self.dModel * FP8 // 8
        len_oscore_weight = self.dModel * self.dInner * FP8 // 8
        len_z = self.seqLen * self.dInner * FP8 // 8
        len_dt_BC = self.seqLen * self.xProjDim * FP8 // 8  # not tiled (FULL dt+BC)
        len_dt_weight_1 = self.dInner * (self.dtRank // self.dtRankUnroll) * self.dConv * FP8 // 8
        len_dt_weight_2 = self.dInner * (self.dtRank // self.dtRankUnroll) * (self.dtRankUnroll - self.dConv) * FP8 // 8
        len_dt_bias = self.dInner * FP8 // 8
        len_x = self.seqLen * self.dInner * FP8 // 8
        len_A = self.dInner * self.dState * FP8 // 8
        len_D = self.dInner * FP8 // 8
        len_y = self.seqLen * self.dInner * FP8 // 8
        len_iscore_weight = self.dModel * self.dInner * FP8 // 8
        len_iscore_out = self.seqLen * self.dModel * BF16 // 8  # not tiled

        nb = self.nb_tiles
        for v in (
            len_oscore_weight,
            len_z,
            len_dt_weight_1,
            len_dt_weight_2,
            len_dt_bias,
            len_x,
            len_A,
            len_D,
            len_y,
            len_iscore_weight,
        ):
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
        # The R10 / R11 start counters are per-kernel-invocation, so they must be
        # computed against the per-tile dInner workload.
        suc_start_cnt, iscore_start_cnt = self.get_safe_to_start_delay(dInner_t)

        tile_scalars = {
            "dInner_tile": dInner_t,
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
            **lengths,
            **deltas,
            **tile_scalars,
            "R10_start_cnt": suc_start_cnt,
            "R11_start_cnt": iscore_start_cnt,
            "dt_to_BC_offset": self.seqLenUnroll * self.dtRank * FP8 // 8,
        }
        self.phase2_scalars = scalars.copy()

        tests = {
            "z": self.seqLen * self.dInner,
            "y": self.seqLen * self.dInner,
            "iscore_out": self.seqLen * self.dModel,
        }

        test_data = {
            name: "uint8_t"
            for name in (
                "oscore_in",
                "oscore_weight",
                "oscore_expected",
                "dt_BC",
                "dt_weight_1",
                "dt_weight_2",
                "dt_bias",
                "suc_A",
                "suc_D",
                "suc_x",
                "suc_expected",
                "iscore_weight",
                "iscore_expected",
            )
        }

        test_data["iscore_bias"] = "uint16_t"

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
