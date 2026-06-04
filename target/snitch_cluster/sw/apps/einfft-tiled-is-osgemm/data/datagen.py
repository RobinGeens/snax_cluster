#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# Datagen for the dual-core (OS + IS in parallel) 2-layer EinFFT MLP.
# Dataflow + rationale: docs/dataflow/06_einfft_mlp.md.
#
# The four per-side matmuls run as two IS_OSGEMM kernel calls (both GEMM cores
# busy each call):
#   REAL side -> OS-core  (rr, ii) : FP8 ConvFormat, fused like einfft.
#   IMAG side -> IS-core  (ri, ir) : raw BF16 flattenCD (NO_REQUANT), simpler fuse.
# No N-tiling: the IS_OSGEMM shared CSRs make OS-N-tile == IS-K-tile, so the
# matmuls run full (L, dPerB) @ (dPerB, dPerB).

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


# Synthesized SIMD mode: SIMD_ADD_BF16 + doRelu (see einfft datagen for the bit-layout sanity check).
_EN_ISCORE_REQUANT_BIT = 1 << 15
_SIMD_MODE_ADD         = 1


def _simd_add_bf16_relu_mode() -> int:
    m_simd = (_SIMD_MODE_ADD << 3) | (1 << 2)
    return _EN_ISCORE_REQUANT_BIT | m_simd


class DataGenerator(DataGeneratorBase):
    APP_NAME = "einfft-tiled-is-osgemm"
    NB_BRANCHES = 4
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
    # Bias expansion helpers.
    # ------------------------------------------------------------------
    def _semi_expand_bias_conv(self, bias_flat, nBranches, dPerB, seqLen, Mu, Nu, conv_unroll):
        """REAL-side bias: conv-walk semi-expanded (mirrors einfft-tiled)."""
        repeat = 16 // conv_unroll
        groups_per_d3 = Nu // conv_unroll
        n_d3 = dPerB // Nu
        l2_count = seqLen // Mu
        assert len(bias_flat) == nBranches * dPerB
        out = []
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
        mode_id = 3  # reuse the OSGEMM (M3_) prefix for all buffers/scalars
        assert f"M{mode_id}_OSGEMM" in self.kwargs
        assert "M4_ISGEMM" in self.kwargs and "M5_ISGEMM_NO_REQUANT" in self.kwargs
        Mu = self.kwargs["seqLenUnroll"]
        Nu = self.kwargs["dInnerUnroll"]
        conv_unroll = self.kwargs["convUnroll"]
        seqLen = self.kwargs["seqLen"]
        dModel = self.kwargs["dModel"]
        oscore_serial_width = self.kwargs["oscore_serial_width"]
        iscore_serial_width = self.kwargs["iscore_serial_width"]
        nBranches = self.NB_BRANCHES

        assert dModel % nBranches == 0
        dPerB = dModel // nBranches
        assert dPerB % Nu == 0, f"dPerB ({dPerB}) must be a multiple of dInnerUnroll ({Nu})"
        # M = seqLen/Mu must be >= 2: the folded imag fuse narrows the IS-core psum directly
        # (no intermediate SIMD-written staging), and the single-seq-tile case (M=1) mis-orders
        # that banked read-back. M>=2 (seqLen >= 2*seqLenUnroll) is verified correct.
        assert seqLen >= 2 * Mu, (
            f"seqLen ({seqLen}) must be >= 2*seqLenUnroll ({2 * Mu}); M=1 breaks the folded imag narrow."
        )

        # ---- Combined IS_OSGEMM mode (OS requant ON, IS requant OFF) ----------
        is_osgemm = self.kwargs["M3_OSGEMM"] | self.kwargs["M4_ISGEMM"]
        requant_bit = self.kwargs["M4_ISGEMM"] ^ self.kwargs["M5_ISGEMM_NO_REQUANT"]
        iosgemm_no_requant = is_osgemm & ~requant_bit
        # Synthesized BF16->FP8 narrow WITH ReLU (doRelu bit) for the imag layer-1 fuse.
        noop_bf16_requant_relu = self.kwargs["M24_SIMD_NOOP_BF16_REQUANT"] | (1 << 2)

        # ============================ OS-core (real side) =====================
        # rr/ii : A=(L,dPerB) flatA, B=(dPerB,dPerB) N_M_K, D=(L,dPerB) ConvFormat.
        a_in_width = Mu * FP8
        b_in_width = self.kwargs["gemm_weight_width"]
        b_array_width = Nu * FP8
        d_array_width = Mu * Nu * FP8
        assert d_array_width % oscore_serial_width == 0
        b_downsize_factor = b_in_width / b_array_width

        M = seqLen // Mu
        K_os = dPerB
        N_os = dPerB // Nu                       # full N (no tiling); requires >= 2
        assert N_os >= 2, f"OSGEMM needs N = dPerB/Nu = {N_os} >= 2; raise dModel"
        downsized_K = int(K_os / b_downsize_factor)
        assert downsized_K == K_os / b_downsize_factor

        os_streamers = {
            "R0": ([K_os, M, N_os], [a_in_width // 8, K_os * a_in_width // 8, 0]),
            "R1": ([downsized_K, M, N_os], [b_in_width // 8, 0, downsized_K * b_in_width // 8]),
            "W0": ([(d_array_width // oscore_serial_width) * M * N_os], [oscore_serial_width // 8]),
        }

        # ============================ IS-core (imag side) =====================
        # ri/ir : A=(L,dPerB) ConvFormat, B=(dPerB,dPerB) K_M_N, CD=(L,dPerB) BF16.
        a_in_width_is = Mu * Nu * FP8
        assert a_in_width_is % iscore_serial_width == 0
        K_t = dPerB // Nu                        # IS reduction unrolled over Nu, full (no tiling)
        downsized_dFinal = int(dPerB / b_downsize_factor)
        assert downsized_dFinal == dPerB / b_downsize_factor

        psum_bounds_and_strides = ([M * dPerB, K_t], [Mu * BF16 // 8, 0])
        is_streamers = {
            "R11": ([K_t * M * (a_in_width_is // iscore_serial_width)], [iscore_serial_width // 8]),
            "R12": ([downsized_dFinal, M, K_t], [b_in_width // 8, 0, downsized_dFinal * b_in_width // 8]),
            "R13": psum_bounds_and_strides,
            "W3":  psum_bounds_and_strides,
        }

        # ============================ SIMD fuse (shared) ======================
        # No tiling -> the fuse operates on the FULL L*dPerB per branch/side.
        n_elems = seqLen * dPerB
        assert n_elems % 32 == 0
        n_fp8_cycles  = n_elems // 32
        n_bf16_cycles = n_elems // 16

        # REAL-side conv-walk bias R13 (4-dim, mirrors einfft full / N_t = N_os).
        groups_per_d3 = Nu // conv_unroll
        group_bytes = 16 * BF16 // 8
        l2_count = seqLen // Mu
        dim1_bound = l2_count * groups_per_d3
        r13_bias_tb = [Mu // (16 // conv_unroll), dim1_bound, N_os, 1]
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

        streamers = {**os_streamers, **is_streamers, **simd_streamers}

        # ============================ Buffer sizes ============================
        len_x_branch    = seqLen * dPerB * FP8 // 8
        len_w_branch    = dPerB * dPerB * FP8 // 8
        len_d           = seqLen * dPerB * FP8 // 8       # OS scratch (rr/ii) ConvFormat FP8
        len_cd          = seqLen * dPerB * BF16 // 8      # IS scratch (ri/ir) BF16
        len_bf16        = seqLen * dPerB * BF16 // 8      # SIMD staging
        len_out_branch  = seqLen * dPerB * FP8 // 8
        len_bias_mini_branch = (dPerB // Nu) * l2_count * groups_per_d3 * 16 * BF16 // 8
        len_bias_im_branch   = seqLen * dPerB * BF16 // 8  # full flattenCD bias

        len_x   = nBranches * len_x_branch
        len_w   = nBranches * len_w_branch
        len_out = nBranches * len_out_branch

        # ---- TCDM footprint sanity check (per-branch resident, L3-staged) ----
        def _a64(v):
            return (v + 63) & ~63

        per_branch_resident = (
            _a64(len_x_branch) * 4 +            # x_re/x_im (flatA) + x_re_conv/x_im_conv (ConvFormat)
            _a64(len_w_branch) * 4 +            # W_re/W_im (OS) + W_re_is/W_im_is (IS)
            _a64(len_d) * 2 +                   # rr, ii (OS scratch)
            _a64(len_cd) * 1 +                  # psum P (IS): seeded with b_im, accumulates ri then ir
            _a64(len_bf16) * 2 +               # bf16_a, bf16_b (real SIMD staging)
            _a64(len_bias_mini_branch) * 1 +   # b_re mini (real)
            _a64(len_out_branch) * 2           # out_re, out_im
        )
        assert per_branch_resident <= self.TCDM_BYTES, (
            f"footprint {per_branch_resident} B exceeds TCDM {self.TCDM_BYTES} B"
        )
        print(f"// einfft-tiled-is-osgemm L1 usage (per-branch resident): {per_branch_resident} B")

        # ---- Biases -----------------------------------------------------------
        # Real: conv-walk semi-expand here. Imag: chisel already emitted the
        # pre-expanded bankTranspose(flattenCD) bias; just pass it through.
        bias_names_re = ("bias_1_real", "bias_2_real")
        bias_re_mini = {}
        for name in bias_names_re:
            flat = self._read_data_int(f"M{mode_id}_{name}.bin")
            bias_re_mini[name] = self._semi_expand_bias_conv(flat, nBranches, dPerB, seqLen, Mu, Nu, conv_unroll)
        bias_im_full = {
            "bias_1_imag": self._read_data_int(f"M{mode_id}_bias_1_imag_bcast.bin"),
            "bias_2_imag": self._read_data_int(f"M{mode_id}_bias_2_imag_bcast.bin"),
        }

        # ---- Specs (DMA'd / verified tensors) --------------------------------
        specs = [
            ("x_real", len_x), ("x_imag", len_x), ("x_real_conv", len_x), ("x_imag_conv", len_x),
            ("x_2_real", len_x), ("x_2_imag", len_x), ("x_2_real_conv", len_x), ("x_2_imag_conv", len_x),
            ("weight_1_real", len_w), ("weight_1_imag", len_w), ("weight_2_real", len_w), ("weight_2_imag", len_w),
            ("weight_1_real_is", len_w), ("weight_1_imag_is", len_w),
            ("weight_2_real_is", len_w), ("weight_2_imag_is", len_w),
            ("output_1_real", len_out), ("output_1_imag", len_out),
            ("output_2_real", len_out), ("output_2_imag", len_out),
        ]
        lengths, deltas = self._collect_lengths_and_deltas(specs)

        scalars = {
            **lengths, **deltas,
            "nBranches":              nBranches,
            "dPerB":                  dPerB,
            "length_x_branch":        len_x_branch,
            "length_w_branch":        len_w_branch,
            "length_d":               len_d,
            "length_cd":              len_cd,
            "length_bf16":            len_bf16,
            "length_out_branch":      len_out_branch,
            "length_bias_mini_branch": len_bias_mini_branch,
            "length_bias_im_branch":  len_bias_im_branch,
            "IS_OSGEMM_NO_REQUANT":   iosgemm_no_requant,
            "SIMD_ADD_BF16_RELU":     _simd_add_bf16_relu_mode(),
            "SIMD_NOOP_BF16_REQUANT_RELU": noop_bf16_requant_relu,
        }

        test_data = {name: "uint8_t" for name, _ in specs}
        tests = {
            "output_1_real": nBranches * seqLen * dPerB,
            "output_1_imag": nBranches * seqLen * dPerB,
            "output_2_real": nBranches * seqLen * dPerB,
            "output_2_imag": nBranches * seqLen * dPerB,
        }

        self.build_mode(mode_id, streamers, scalars=scalars, test_data=test_data, tests=tests)

        for name in bias_names_re:
            self.format_vector("uint16_t", f"M{mode_id}_{name}_mini", bias_re_mini[name])
        for name in ("bias_1_imag", "bias_2_imag"):
            self.format_vector("uint16_t", f"M{mode_id}_{name}_full", bias_im_full[name])


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
