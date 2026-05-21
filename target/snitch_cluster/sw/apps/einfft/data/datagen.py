#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Un-tiled OS-core 2-layer complex MLP (the post-EinFFT MLP block).
#
# Per (layer, branch) we run 4 OSGEMMs and one SIMD fuse pipeline per side
# (real, imag). Branches are walked serially.
#
# This datagen handles ALL of the ConvFormat / OS-core / SIMD geometry so the
# C side does no byte-permutation work:
#   - The chisel reference now emits `output_*_real|imag` in ConvFormat per
#     branch (= the same byte order the OS-core writes), so the SNAX program
#     can verify TCDM directly.
#   - `x_2_real|imag` is emitted in flatA per branch (= the OS-core A layout)
#     so layer 2's A input is DMA'd straight in without any reformat.
#   - `bias_*_*_bcast` is pre-expanded here: for each (branch, ConvFormat
#     element position t) we store `bias[col(t)]`, so the SIMD ADD step
#     reads it linearly with R13 alongside the linear-walked bf16_a buffer.
#
# OSGEMM streamer config + per-SIMD-step R7/R13/W3 configs are also emitted
# here. The C side just calls `set_*_streamer_csr` with the right constants.

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


# Synthesized SIMD mode value. The Chisel SimbaCoreMode bit layout is
# (high to low): boolean enables · m_switchCoreMode · sw_simdInType · m_simd,
# where m_simd = (SimdMode<<3) | (doRelu<<2) | (doRequant<<1) | doSoftShrink.
# `en_isCoreRequant` defaults to 1 → bit 15 is set on every plain SIMD mode.
# Sanity-check: M8_SIMD_ADD_BF16 = 32776 = 0x8008 = (en_isCoreRequant) | (Add<<3).
_EN_ISCORE_REQUANT_BIT = 1 << 15
_SIMD_MODE_ADD = 1


def _simd_add_bf16_relu_mode() -> int:
    """ADD_BF16 with doRelu=1 — used so layer-1's bias-ADD also applies ReLU."""
    m_simd = (_SIMD_MODE_ADD << 3) | (1 << 2)   # mode=Add, doRelu=1
    return _EN_ISCORE_REQUANT_BIT | m_simd      # 0x800c = 32780


class DataGenerator(DataGeneratorBase):
    APP_NAME = "einfft"
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
        """Yield (row, col) for each conv-walk linear position in a (L, dPerB) tensor.

        Loop order (outermost→innermost): d3 (tile-col), l2 (tile-row),
        d2 (sub-tile col), l1, c. For L=Mu only one l2 iteration.
        """
        assert L % Mu == 0 and dPerB % Nu == 0 and Nu % conv_unroll == 0
        for d3 in range(dPerB // Nu):
            for l2 in range(L // Mu):
                for d2 in range(Nu // conv_unroll):
                    for l1 in range(Mu):
                        for c in range(conv_unroll):
                            yield l2 * Mu + l1, d3 * Nu + d2 * conv_unroll + c

    def _expand_bias_conv_order(self, bias_flat: list[int], nBranches: int, L: int, dPerB: int,
                                Mu: int, Nu: int, conv_unroll: int) -> list[int]:
        """Pre-stage bias for R13 broadcast: for each (branch, conv-walk position t),
        emit bias[branch][col(t)]. main.c's SIMD ADD reads this linearly alongside
        bf16_a (which is in conv-walk byte order after the linear widen)."""
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
        oscore_serial_width = self.kwargs["oscore_serial_width"]
        nBranches = self.NB_BRANCHES

        assert dModel % nBranches == 0, f"dModel ({dModel}) must be divisible by nBranches ({nBranches})"
        dPerB = dModel // nBranches

        a_in_width = Mu * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = Nu * FP8
        d_array_width = Mu * Nu * FP8
        assert d_array_width % oscore_serial_width == 0, "d_array_width must be divisible by oscore_serial_width"
        b_downsize_factor = b_in_width / b_array_width  # >= 1

        # OS-core MatmulDims (per branch, per matmul): A=(L, dPerB), B=(dPerB, dPerB), D=(L, dPerB)
        M = seqLen // Mu                                  # 1 for L=Mu
        K = dPerB
        N = dPerB // Nu
        downsized_K = int(K / b_downsize_factor)
        assert downsized_K == K / b_downsize_factor, f"downsized_K {K / b_downsize_factor} must be an integer"
        assert downsized_K * b_in_width == K * b_array_width

        n_elems  = seqLen * dPerB                         # per-side full element count

        # ---- OSGEMM streamer config (one matmul; only R0/R1/W0 base ptrs change) ----
        osgemm_streamers = {
            "R0": (
                [K, M, N],
                [a_in_width // 8, K * a_in_width // 8, 0],
            ),
            "R1": (
                [downsized_K, M, N],
                [b_in_width // 8, 0, downsized_K * b_in_width // 8],
            ),
            "W0": (
                [(d_array_width // oscore_serial_width) * M * N],
                [oscore_serial_width // 8],
            ),
        }

        # ---- SIMD streamer configs ---------------------------------------------
        # bf16 staging buffers are walked LINEARLY in conv-walk byte order (same
        # byte order the OS-core W0 wrote rr/ii/ri/ir, just doubled width after
        # widen). Lane k of R7 / R13 / W3 pairs at the same logical conv-walk
        # element position — see docs/memory_layouts/10_simd.md §10.2 / §10.8.
        n_fp8_cycles  = n_elems // 32                     # FP8: 32 lanes/cycle
        n_bf16_cycles = n_elems // 16                     # BF16: 16 lanes/cycle

        simd_streamers = {
            # Widen FP8 → BF16.
            "R7_widen": ([n_fp8_cycles,  1, 1, 1], [32, 0, 0, 0], [BANK_BYTES, 2 * BANK_BYTES]),
            "W3_widen": ([n_bf16_cycles, 1, 1, 1], [32, 0, 0, 0], [BANK_BYTES]),
            # BF16 binop (ADD / SUB).
            "R7_bf16":  ([n_bf16_cycles, 1, 1, 1], [32, 0, 0, 0], [BANK_BYTES, 2 * BANK_BYTES]),
            "R13_bf16": ([n_bf16_cycles, 1, 1, 1], [32, 0, 0, 0], [BANK_BYTES]),
            "W3_bf16":  ([n_bf16_cycles, 1, 1, 1], [32, 0, 0, 0], [BANK_BYTES]),
            # Narrow BF16 → FP8.
            "W3_fp8":   ([n_fp8_cycles,  1, 1, 1], [32, 0, 0, 0], [BANK_BYTES]),
        }

        streamers: dict = {**osgemm_streamers, **simd_streamers}

        # ---- Buffer sizes -----------------------------------------------------
        len_x_branch    = seqLen * dPerB * FP8 // 8
        len_out_branch  = seqLen * dPerB * FP8 // 8
        len_w_branch    = dPerB * dPerB * FP8 // 8
        len_bias_branch = dPerB * BF16 // 8
        len_d           = seqLen * dPerB * FP8 // 8
        len_bf16        = n_elems * BF16 // 8
        len_bias_bcast_branch = n_elems * BF16 // 8       # per branch, conv-walk-order

        # Concatenated-over-branches lengths.
        len_x          = nBranches * len_x_branch
        len_w          = nBranches * len_w_branch
        len_bias       = nBranches * len_bias_branch
        len_bias_bcast = nBranches * len_bias_bcast_branch
        len_out        = nBranches * len_out_branch

        # ---- TCDM footprint sanity check --------------------------------------
        def _align64(v: int) -> int:
            return (v + 63) & ~63

        shared_bytes = (
            len_x + len_x +                                # x_real, x_imag (layer 1)
            len_x + len_x +                                # x_2_real, x_2_imag (layer 2)
            len_out + len_out +                            # l1_real, l1_imag (verify against ConvFormat ref)
            len_out + len_out +                            # l2_real, l2_imag
            len_bias_bcast * 4                             # b1_re/im, b2_re/im (pre-expanded)
        )
        per_branch_bytes = (
            _align64(len_w_branch) * 2 +                   # W_re + W_im
            _align64(len_d) * 4 +                          # rr, ii, ri, ir
            _align64(len_bf16) * 2                         # bf16_a, bf16_b
        )
        total_l1_bytes = _align64(shared_bytes) + per_branch_bytes
        assert total_l1_bytes <= self.TCDM_BYTES, (
            f"einfft L1 footprint {total_l1_bytes} B exceeds TCDM budget {self.TCDM_BYTES} B."
        )
        print(f"// einfft L1 usage: {total_l1_bytes} B "
              f"(shared {_align64(shared_bytes)} B, per-branch {per_branch_bytes} B)")

        # ---- Pre-expand the bias buffers --------------------------------------
        # The chisel datagen writes bias as nBranches × dPerB BF16 ints (one int
        # per line) into M3_bias_<X>_<re|im>.bin. We expand to conv-walk order
        # and emit as a new uint16_t array DMA'd straight into TCDM.
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

        # ---- Specs: register tensor lengths + L3 base addresses for main.c. ----
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
            "nBranches":              nBranches,
            "dPerB":                  dPerB,
            "length_x_branch":        len_x_branch,
            "length_out_branch":      len_out_branch,
            "length_w_branch":        len_w_branch,
            "length_bias_branch":     len_bias_branch,
            "length_d":               len_d,
            "length_bf16":            len_bf16,
            "length_bias_bcast_branch": len_bias_bcast_branch,
            "SIMD_ADD_BF16_RELU":     _simd_add_bf16_relu_mode(),
        }

        # ---- Test data + sampled checks ---------------------------------------
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

        # Emit the pre-expanded bias buffers (no underlying .bin file; computed above).
        for name in bias_names:
            self.format_vector("uint16_t", f"M{mode_id}_{name}_bcast", bias_expanded[name])


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
