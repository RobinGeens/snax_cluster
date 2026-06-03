#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Datagen for the N-tiled, per-tile-fuse OS-core 2-layer EinFFT MLP.
# Dataflow, tiling rationale, and TCDM footprint: docs/dataflow/06_einfft_mlp.md.
#
# Emits per-tile OSGEMM streamer config (N_t = N_full / nb_tiles), per-tile
# SIMD streamer bounds (n_elems = L * dPerB_t), and the mini-expanded bias
# (conv-walk order) sliced per tile via length_bias_mini_tile.

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


# Synthesized SIMD mode: SIMD_ADD_BF16 + doRelu. See einfft/datagen.py for the
# SimbaCoreMode bit-layout sanity check (verified against M8_SIMD_ADD_BF16).
_EN_ISCORE_REQUANT_BIT = 1 << 15
_SIMD_MODE_ADD         = 1


def _simd_add_bf16_relu_mode() -> int:
    m_simd = (_SIMD_MODE_ADD << 3) | (1 << 2)
    return _EN_ISCORE_REQUANT_BIT | m_simd


class DataGenerator(DataGeneratorBase):
    APP_NAME = "einfft-tiled"
    NB_BRANCHES = 4

    # TCDM budget — keep in sync with snax_simbacore_cluster.hjson (tcdm.size = 512 kB).
    TCDM_BYTES = 512 * 1024

    def __init__(self, **kwargs):
        super().__init__(self.APP_NAME, **kwargs)
        local_params_path = pathlib.Path(__file__).resolve().parent / "params_in.hjson"
        with local_params_path.open() as f:
            local_params = hjson.loads(f.read())
        for key, value in local_params.items():
            self.kwargs.setdefault(key, value)

    def run(self):
        self.build_einfft_data()
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

    # ------------------------------------------------------------------
    # ConvFormat helpers (Python-side; mirror chisel-ssm/MambaCoreUtil).
    # ------------------------------------------------------------------
    @staticmethod
    def _conv_walk_indices(L: int, dPerB: int, Mu: int, Nu: int, conv_unroll: int):
        """Yield (row, col) for each conv-walk linear position in a (L, dPerB)
        tensor (loop order: d3 outer, l2, d2, l1, c inner)."""
        assert L % Mu == 0 and dPerB % Nu == 0 and Nu % conv_unroll == 0
        for d3 in range(dPerB // Nu):
            for l2 in range(L // Mu):
                for d2 in range(Nu // conv_unroll):
                    for l1 in range(Mu):
                        for c in range(conv_unroll):
                            yield l2 * Mu + l1, d3 * Nu + d2 * conv_unroll + c

    def _semi_expand_bias(self, bias_flat: list[int], nBranches: int, dPerB: int,
                          seqLen: int, Mu: int, Nu: int, conv_unroll: int) -> list[int]:
        """Semi-expand bias: repeat each conv_unroll-wide column group
        (16/conv_unroll) times per BF16 cycle, and duplicate d2 groups L/Mu
        times per d3 block so R13 walks flat (no negative stride).
        Size: (dPerB/Nu)*(L/Mu)*(Nu/cu)*32 bytes/branch."""
        repeat = 16 // conv_unroll
        groups_per_d3 = Nu // conv_unroll
        n_d3 = dPerB // Nu
        l2_count = seqLen // Mu
        assert len(bias_flat) == nBranches * dPerB
        out: list[int] = []
        for branch in range(nBranches):
            bb = bias_flat[branch * dPerB:(branch + 1) * dPerB]
            for d3 in range(n_d3):
                for _ in range(l2_count):
                    for d2 in range(groups_per_d3):
                        g = d3 * groups_per_d3 + d2
                        group = bb[g * conv_unroll:(g + 1) * conv_unroll]
                        out.extend(group * repeat)
        expected = nBranches * n_d3 * l2_count * groups_per_d3 * 16
        assert len(out) == expected, f"{len(out)} != {expected}"
        return out

    def build_einfft_data(self):
        mode_id = 3  # OSGEMM
        assert f"M{mode_id}_OSGEMM" in self.kwargs, f"verify mode_id {mode_id} for OSGEMM"
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        conv_unroll = self.kwargs["convUnroll"]
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        nb_tiles = self.kwargs["nb_tiles"]
        oscore_serial_width = self.kwargs["oscore_serial_width"]
        nBranches = self.NB_BRANCHES

        assert dModel % nBranches == 0, f"dModel ({dModel}) must be divisible by nBranches ({nBranches})"
        dPerB = dModel // nBranches

        a_in_width = Mu * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = Nu * FP8
        d_array_width = Mu * Nu * FP8
        assert d_array_width % oscore_serial_width == 0, "d_array_width must be divisible by oscore_serial_width"
        b_downsize_factor = b_in_width / b_array_width

        # OS-core MatmulDims (per branch, per matmul): A=(L, dPerB), B=(dPerB, dPerB), D=(L, dPerB)
        M = seqLen // Mu                                  # 1 for L = Mu
        K = dPerB
        N_full = dPerB // Nu                              # 2 for default
        downsized_K = int(K / b_downsize_factor)
        assert downsized_K == K / b_downsize_factor
        assert downsized_K * b_in_width == K * b_array_width

        # Tile along N (output-channel axis).
        assert N_full % nb_tiles == 0, (
            f"N ({N_full}) must be divisible by nb_tiles ({nb_tiles}); equivalently, "
            f"dPerB ({dPerB}) must be a multiple of dInnerUnroll * nb_tiles ({Nu} * {nb_tiles})"
        )
        N_t = N_full // nb_tiles
        # OSGEMM requires N >= 2 (see chisel-ssm/docs/memory_layouts/02_gemm_layouts.md §2.4).
        assert N_t >= 2, (
            f"N_t = N_full / nb_tiles = {N_full} / {nb_tiles} = {N_t} < 2. "
            f"OSGEMM requires N >= 2; for dPerB={dPerB} the cap is nb_tiles <= {N_full // 2}."
        )
        dPerB_t = dPerB // nb_tiles
        n_elems_tile = seqLen * dPerB_t                   # SIMD now operates per-tile (one N-tile at a time)
        assert n_elems_tile % 32 == 0

        # ---- OSGEMM streamer config (one per-tile matmul) --------------------
        osgemm_streamers = {
            "R0": (
                [K, M, N_t],
                [a_in_width // 8, K * a_in_width // 8, 0],
            ),
            "R1": (
                [downsized_K, M, N_t],
                [b_in_width // 8, 0, downsized_K * b_in_width // 8],
            ),
            "W0": (
                [(d_array_width // oscore_serial_width) * M * N_t],
                [oscore_serial_width // 8],
            ),
        }

        # ---- SIMD streamer configs (PER-TILE: one N-tile per fuse) ----------
        # The fuse now runs once per (side, tile) at tile bounds (n_elems_tile),
        # so the OS-core scratch / BF16 staging / output only need to hold ONE
        # tile. The old "short bounds under-run" caveat applies only to tiny
        # absolute bounds (~12 fp8 cycles); per-tile bounds at realistic params
        # are far above that. See docs/dataflow/06_einfft_mlp.md §6.4.
        n_fp8_cycles  = n_elems_tile // 32
        n_bf16_cycles = n_elems_tile // 16

        # R13 bias: 4-dim walk over mini-expanded bias (per tile = N_t d3 blocks).
        # dim0: l1-block repeat (stride=0 → Reader repeater handles it)
        # dim1: l2 repeat (stride=0 → AGU natively stays; NOT innermost, no workaround)
        # dim2: d2 group advance
        # dim3: d3 block advance (only N_t blocks belong to this tile)
        groups_per_d3 = Nu // conv_unroll       # 6
        group_bytes = 16 * BF16 // 8            # 32
        l2_count = seqLen // Mu
        dim1_bound = l2_count * groups_per_d3
        r13_bias_tb = [Mu // (16 // conv_unroll),  # 4: l1-block repeat (repeater)
                       dim1_bound,                  # l2*d2 flat walk within d3
                       N_t,                         # d3 blocks in ONE tile
                       1]
        r13_bias_ts = [0, group_bytes, dim1_bound * group_bytes, 0]

        simd_streamers = {
            "R7_widen": ([n_fp8_cycles,  1, 1, 1], [32, 0, 0, 0], [BANK_BYTES, 2 * BANK_BYTES]),
            "W3_widen": ([n_bf16_cycles, 1, 1, 1], [32, 0, 0, 0], [BANK_BYTES]),
            "R7_bf16":  ([n_bf16_cycles, 1, 1, 1], [32, 0, 0, 0], [BANK_BYTES, 2 * BANK_BYTES]),
            "R13_bf16": ([n_bf16_cycles, 1, 1, 1], [32, 0, 0, 0], [BANK_BYTES]),
            "R13_bias": (r13_bias_tb, r13_bias_ts, [BANK_BYTES]),
            "W3_bf16":  ([n_bf16_cycles, 1, 1, 1], [32, 0, 0, 0], [BANK_BYTES]),
            "W3_fp8":   ([n_fp8_cycles,  1, 1, 1], [32, 0, 0, 0], [BANK_BYTES]),
        }

        streamers: dict = {**osgemm_streamers, **simd_streamers}

        # ---- Buffer sizes -----------------------------------------------------
        len_x_branch    = seqLen * dPerB * FP8 // 8
        len_out_branch  = seqLen * dPerB * FP8 // 8
        len_w_branch    = dPerB * dPerB * FP8 // 8
        assert len_w_branch % nb_tiles == 0
        len_w_branch_tile = len_w_branch // nb_tiles
        len_bias_branch = dPerB * BF16 // 8
        len_d_tile      = seqLen * dPerB_t * FP8 // 8     # one per-tile OS-core scratch slot
        len_bf16        = n_elems_tile * BF16 // 8        # per-tile BF16 staging
        len_bias_mini_branch = (dPerB // Nu) * l2_count * groups_per_d3 * 16 * BF16 // 8
        assert len_bias_mini_branch % nb_tiles == 0
        len_bias_mini_tile = len_bias_mini_branch // nb_tiles  # contiguous d3-major slice per tile

        len_x          = nBranches * len_x_branch
        len_w          = nBranches * len_w_branch
        len_bias       = nBranches * len_bias_branch
        len_bias_mini  = nBranches * len_bias_mini_branch
        len_out        = nBranches * len_out_branch

        # ---- TCDM footprint sanity check --------------------------------------
        # (L3-staged) layout: only the CURRENT branch's slots live in TCDM.
        # x/bias_bcast/output DMA per (layer, branch); 4 output arrays
        # (l1_re/im, l2_re/im) live in L3 (snrt_l3alloc).
        def _align64(v: int) -> int:
            return (v + 63) & ~63

        per_branch_resident = (
            _align64(len_x_branch) * 2 +                    # x_re_b + x_im_b (FULL: reused by every N-tile)
            _align64(len_bias_mini_branch) * 2 * 2 +        # b_re_pp[2] + b_im_pp[2] (FULL per branch)
            _align64(len_d_tile) * 2 * 2 +                  # out_re_pp[2] + out_im_pp[2] (TILE, ping-pong)
            _align64(len_w_branch_tile) * 2 * 2 +           # W_re_pp[2] + W_im_pp[2]
            _align64(len_d_tile) * 4 +                      # rr, ii, ri, ir (TILE)
            _align64(len_bf16) * 2                          # bf16_a, bf16_b (TILE)
        )
        total_l1_bytes = per_branch_resident
        assert total_l1_bytes <= self.TCDM_BYTES, (
            f"einfft-tiled L1 footprint {total_l1_bytes} B exceeds TCDM budget {self.TCDM_BYTES} B."
        )
        print(f"// einfft-tiled L1 usage (per-branch resident): {total_l1_bytes} B")

        # ---- Mini-expand bias per branch for streamer reuse -------------------
        bias_names = ("bias_1_real", "bias_1_imag", "bias_2_real", "bias_2_imag")
        bias_mini: dict[str, list[int]] = {}
        for name in bias_names:
            try:
                flat = self._read_data_int(f"M{mode_id}_{name}.bin")
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Missing chisel-ssm output for {name}: {e}. Run the scala data generator first."
                )
            bias_mini[name] = self._semi_expand_bias(flat, nBranches, dPerB, seqLen, Mu, Nu, conv_unroll)

        # ---- Specs ------------------------------------------------------------
        specs = [
            ("x_real",        len_x),
            ("x_imag",        len_x),
            ("x_2_real",      len_x),
            ("x_2_imag",      len_x),
            ("weight_1_real", len_w),
            ("weight_1_imag", len_w),
            ("weight_2_real", len_w),
            ("weight_2_imag", len_w),
            ("bias_1_real_mini", len_bias_mini),
            ("bias_1_imag_mini", len_bias_mini),
            ("bias_2_real_mini", len_bias_mini),
            ("bias_2_imag_mini", len_bias_mini),
            ("output_1_real", len_out),
            ("output_1_imag", len_out),
            ("output_2_real", len_out),
            ("output_2_imag", len_out),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)

        scalars = {
            **lengths,
            **deltas,
            "nBranches":                 nBranches,
            "dPerB":                     dPerB,
            "dPerB_tile":                dPerB_t,
            "length_x_branch":           len_x_branch,
            "length_out_branch":         len_out_branch,
            "length_w_branch":           len_w_branch,
            "length_w_branch_tile":      len_w_branch_tile,
            "length_d_tile":             len_d_tile,
            "length_bf16":               len_bf16,
            "length_bias_mini_branch":  len_bias_mini_branch,
            "length_bias_mini_tile":    len_bias_mini_tile,
            "SIMD_ADD_BF16_RELU":        _simd_add_bf16_relu_mode(),
        }

        test_data = {
            name: "uint8_t"
            for name in (
                "x_real",
                "x_imag",
                "x_2_real",
                "x_2_imag",
                "weight_1_real",
                "weight_1_imag",
                "weight_2_real",
                "weight_2_imag",
                "output_1_real",
                "output_1_imag",
                "output_2_real",
                "output_2_imag",
            )
        }

        tests = {
            "output_1_real": nBranches * seqLen * dPerB,
            "output_1_imag": nBranches * seqLen * dPerB,
            "output_2_real": nBranches * seqLen * dPerB,
            "output_2_imag": nBranches * seqLen * dPerB,
        }

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)

        for name in bias_names:
            self.format_vector("uint16_t", f"M{mode_id}_{name}_mini", bias_mini[name])


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
