#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Tiled, double-buffered OS-core 2-layer EinFFT MLP.
#
# Same ConvFormat-throughout strategy as the un-tiled `einfft` (see
# docs/dataflow/06_einfft_mlp.md). The N (= D/4 output-channel) axis
# is tiled at the OS-core level only: weights are DMA'd one tile at a time
# (ping-pong against compute) and the per-tile OSGEMM output drops into the
# right slot of the FULL per-branch ConvFormat scratch. The SIMD fuse then
# runs ONCE per side per branch over the assembled scratch — keeping the
# SIMD launch bounds at FULL per-branch size (the un-tiled config), which
# is what the hardware empirically handles correctly.
#
# This datagen therefore emits:
#   - OSGEMM streamer config with N_t = N_full / nb_tiles (per-tile matmul).
#   - SIMD streamer configs with n_elems = L * dPerB (per-branch / un-tiled).
#   - Bias broadcast pre-expanded to full per-branch conv-walk order
#     (same layout as `einfft`).

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

    def _expand_bias_conv_order(self, bias_flat: list[int], nBranches: int, L: int, dPerB: int,
                                Mu: int, Nu: int, conv_unroll: int) -> list[int]:
        """Per (branch, conv-walk position t over the full (L, dPerB) tensor),
        emit `bias[branch][col(t)]`. The SIMD ADD step's R13 walks this linearly
        alongside the full per-branch bf16_a buffer."""
        assert len(bias_flat) == nBranches * dPerB
        out: list[int] = []
        for branch in range(nBranches):
            branch_bias = bias_flat[branch * dPerB:(branch + 1) * dPerB]
            for _, col in self._conv_walk_indices(L, dPerB, Mu, Nu, conv_unroll):
                out.append(branch_bias[col])
        assert len(out) == nBranches * L * dPerB
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
        n_elems_full = seqLen * dPerB                     # SIMD operates on FULL per-branch buffers
        assert n_elems_full % 32 == 0

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

        # ---- SIMD streamer configs (FULL per-branch, NOT per-tile) ----------
        # Empirically the SIMD widen / narrow paths under-run on short bounds
        # (e.g. 12 fp8 cycles → wrong output), so we keep the bounds at the
        # full per-branch size and feed the SIMD a pre-assembled FULL scratch.
        n_fp8_cycles  = n_elems_full // 32
        n_bf16_cycles = n_elems_full // 16

        simd_streamers = {
            "R7_widen": ([n_fp8_cycles,  1, 1, 1], [32, 0, 0, 0], [BANK_BYTES, 2 * BANK_BYTES]),
            "W3_widen": ([n_bf16_cycles, 1, 1, 1], [32, 0, 0, 0], [BANK_BYTES]),
            "R7_bf16":  ([n_bf16_cycles, 1, 1, 1], [32, 0, 0, 0], [BANK_BYTES, 2 * BANK_BYTES]),
            "R13_bf16": ([n_bf16_cycles, 1, 1, 1], [32, 0, 0, 0], [BANK_BYTES]),
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
        len_d_branch    = len_out_branch                  # FULL per-branch scratch (= sum of tiles)
        len_bf16        = n_elems_full * BF16 // 8        # FULL per-branch BF16 staging
        len_bias_bcast_branch = n_elems_full * BF16 // 8  # FULL per-branch bias broadcast

        len_x          = nBranches * len_x_branch
        len_w          = nBranches * len_w_branch
        len_bias       = nBranches * len_bias_branch
        len_bias_bcast = nBranches * len_bias_bcast_branch
        len_out        = nBranches * len_out_branch

        # ---- TCDM footprint sanity check --------------------------------------
        def _align64(v: int) -> int:
            return (v + 63) & ~63

        shared_bytes = (
            len_x + len_x +                                # x_real, x_imag
            len_x + len_x +                                # x_2_real, x_2_imag
            len_out + len_out +                            # l1_real, l1_imag
            len_out + len_out +                            # l2_real, l2_imag
            len_bias_bcast * 4                             # 4 bias_bcast buffers
        )
        per_branch_bytes = (
            _align64(len_w_branch_tile) * 2 * 2 +          # W_re_pp[2] + W_im_pp[2]
            _align64(len_d_branch) * 4 +                   # rr, ii, ri, ir (FULL)
            _align64(len_bf16) * 2                         # bf16_a, bf16_b
        )
        total_l1_bytes = _align64(shared_bytes) + per_branch_bytes
        assert total_l1_bytes <= self.TCDM_BYTES, (
            f"einfft-tiled L1 footprint {total_l1_bytes} B exceeds TCDM budget {self.TCDM_BYTES} B."
        )
        print(f"// einfft-tiled L1 usage: {total_l1_bytes} B "
              f"(shared {_align64(shared_bytes)} B, per-branch {per_branch_bytes} B)")

        # ---- Pre-expand bias per branch in FULL conv-walk order --------------
        bias_names = ("bias_1_real", "bias_1_imag", "bias_2_real", "bias_2_imag")
        bias_expanded: dict[str, list[int]] = {}
        for name in bias_names:
            try:
                flat = self._read_data_int(f"M{mode_id}_{name}.bin")
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Missing chisel-ssm output for {name}: {e}. Run the scala data generator first."
                )
            bias_expanded[name] = self._expand_bias_conv_order(
                flat, nBranches, seqLen, dPerB, Mu, Nu, conv_unroll
            )

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
            ("bias_1_real_bcast", len_bias_bcast),
            ("bias_1_imag_bcast", len_bias_bcast),
            ("bias_2_real_bcast", len_bias_bcast),
            ("bias_2_imag_bcast", len_bias_bcast),
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
            "length_bias_bcast_branch": len_bias_bcast_branch,
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
            self.format_vector("uint16_t", f"M{mode_id}_{name}_bcast", bias_expanded[name])


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
