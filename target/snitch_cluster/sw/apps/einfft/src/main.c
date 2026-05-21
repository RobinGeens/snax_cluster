// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Un-tiled 2-layer complex MLP that follows the EinFFT block (see
// docs/dataflow/06_einfft_complex_mlp.md). Every per-side intermediate is
// kept in the OS-core's native ConvFormat byte order — the chisel reference
// emits `output_*_*` in the same layout, and the bias is pre-expanded by
// datagen.py to match — so there is no byte permutation on the C side.
//
// Per branch, per layer:
//   4 OSGEMMs (rr / ir / ii / ri, in ConvFormat)
//   per side:
//     widen pos (FP8 → BF16 NOOP_FP8_REQUANT)
//     widen neg
//     bf16 binop (SUB on real side, ADD on imag)
//     bf16 ADD with bias broadcast (layer 1 also applies ReLU via doRelu)
//     narrow bf16 → FP8 NOOP_BF16_REQUANT, write straight to ConvFormat output
//
// All SIMD streamer strides come from datagen.py (M3_R*, M3_W*); only the
// base pointers change between launches.

#include "data.h"
#include "snax-simbacore-lib.h"

int main(void) {
    int err = 0;

    void* tcdm_base_ptr = snrt_l1_next();

    uint8_t* ptr_x_re      = (uint8_t*)(tcdm_base_ptr + M3_addr_x_real);
    uint8_t* ptr_x_im      = (uint8_t*)(tcdm_base_ptr + M3_addr_x_imag);
    uint8_t* ptr_x2_re     = (uint8_t*)(tcdm_base_ptr + M3_addr_x_2_real);
    uint8_t* ptr_x2_im     = (uint8_t*)(tcdm_base_ptr + M3_addr_x_2_imag);
    uint8_t* ptr_l1_re     = (uint8_t*)(tcdm_base_ptr + M3_addr_output_1_real);
    uint8_t* ptr_l1_im     = (uint8_t*)(tcdm_base_ptr + M3_addr_output_1_imag);
    uint8_t* ptr_l2_re     = (uint8_t*)(tcdm_base_ptr + M3_addr_output_2_real);
    uint8_t* ptr_l2_im     = (uint8_t*)(tcdm_base_ptr + M3_addr_output_2_imag);
    uint16_t* ptr_b1_re_bc = (uint16_t*)(tcdm_base_ptr + M3_addr_bias_1_real_bcast);
    uint16_t* ptr_b1_im_bc = (uint16_t*)(tcdm_base_ptr + M3_addr_bias_1_imag_bcast);
    uint16_t* ptr_b2_re_bc = (uint16_t*)(tcdm_base_ptr + M3_addr_bias_2_real_bcast);
    uint16_t* ptr_b2_im_bc = (uint16_t*)(tcdm_base_ptr + M3_addr_bias_2_imag_bcast);

    // Per-branch scratch / staging — placed right after the FULL inputs.
    uint8_t* ptr_W_re    = (uint8_t*)(tcdm_base_ptr + M3_addr_output_2_imag + M3_length_output_2_imag);
    uint8_t* ptr_W_im    = ptr_W_re + M3_length_w_branch;
    uint8_t* ptr_rr      = ptr_W_im + M3_length_w_branch;
    uint8_t* ptr_ii      = ptr_rr + M3_length_d;
    uint8_t* ptr_ri      = ptr_ii + M3_length_d;
    uint8_t* ptr_ir      = ptr_ri + M3_length_d;
    uint16_t* ptr_bf16_a = (uint16_t*)(ptr_ir + M3_length_d);
    uint16_t* ptr_bf16_b = (uint16_t*)((uint8_t*)ptr_bf16_a + M3_length_bf16);

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_x_re, M3_x_real, M3_length_x_real);
        snrt_dma_start_1d(ptr_x_im, M3_x_imag, M3_length_x_imag);
        snrt_dma_start_1d(ptr_x2_re, M3_x_2_real, M3_length_x_2_real);
        snrt_dma_start_1d(ptr_x2_im, M3_x_2_imag, M3_length_x_2_imag);
        snrt_dma_start_1d((uint8_t*)ptr_b1_re_bc, (uint8_t*)M3_bias_1_real_bcast, M3_length_bias_1_real_bcast);
        snrt_dma_start_1d((uint8_t*)ptr_b1_im_bc, (uint8_t*)M3_bias_1_imag_bcast, M3_length_bias_1_imag_bcast);
        snrt_dma_start_1d((uint8_t*)ptr_b2_re_bc, (uint8_t*)M3_bias_2_real_bcast, M3_length_bias_2_real_bcast);
        snrt_dma_start_1d((uint8_t*)ptr_b2_im_bc, (uint8_t*)M3_bias_2_imag_bcast, M3_length_bias_2_imag_bcast);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t simbacore_cycles = 0, start_cycles = 0;
    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: einfft (un-tiled) "
            "(L=%u, dModel=%u, dPerB=%u, nBranches=%u)\n\n",
            seqLen, dModel, M3_dPerB, M3_nBranches);
        start_cycles = snrt_mcycle();
    }

    // ==================================================================
    // Two-layer loop. Each layer is structurally identical; only the
    // input pointers, weight tensors, bias broadcasts and the layer-1
    // ReLU mode differ.
    // ==================================================================
    for (int layer = 0; layer < 2; layer++) {
        uint8_t* layer_x_re     = (layer == 0) ? ptr_x_re : ptr_x2_re;
        uint8_t* layer_x_im     = (layer == 0) ? ptr_x_im : ptr_x2_im;
        uint8_t* layer_out_re   = (layer == 0) ? ptr_l1_re : ptr_l2_re;
        uint8_t* layer_out_im   = (layer == 0) ? ptr_l1_im : ptr_l2_im;
        uint8_t* layer_w_re_l3  = (layer == 0) ? M3_weight_1_real : M3_weight_2_real;
        uint8_t* layer_w_im_l3  = (layer == 0) ? M3_weight_1_imag : M3_weight_2_imag;
        uint16_t* layer_b_re_bc = (layer == 0) ? ptr_b1_re_bc : ptr_b2_re_bc;
        uint16_t* layer_b_im_bc = (layer == 0) ? ptr_b1_im_bc : ptr_b2_im_bc;
        uint32_t add_bias_mode  = (layer == 0) ? M3_SIMD_ADD_BF16_RELU : M8_SIMD_ADD_BF16;

        for (uint32_t b = 0; b < M3_nBranches; b++) {
            // DMA this branch's W_re / W_im into the resident TCDM slot.
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_W_re, layer_w_re_l3 + b * M3_length_w_branch, M3_length_w_branch);
                snrt_dma_start_1d(ptr_W_im, layer_w_im_l3 + b * M3_length_w_branch, M3_length_w_branch);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();

            if (snrt_global_core_idx() == 0) {
                uint8_t* x_re_b     = layer_x_re + b * M3_length_x_branch;
                uint8_t* x_im_b     = layer_x_im + b * M3_length_x_branch;
                uint8_t* out_re_b   = layer_out_re + b * M3_length_out_branch;
                uint8_t* out_im_b   = layer_out_im + b * M3_length_out_branch;
                uint16_t* b_re_bc_b = layer_b_re_bc + b * (M3_length_bias_bcast_branch / 2);
                uint16_t* b_im_bc_b = layer_b_im_bc + b * (M3_length_bias_bcast_branch / 2);

                // ---------------- 4 OSGEMMs ----------------------------------
                // rr = x_re @ W_re; ir = x_im @ W_re; ii = x_im @ W_im; ri = x_re @ W_im.
                set_osgemm_streamer_csr((uint32_t)x_re_b, M3_R0_ss, M3_R0_tb, M3_R0_ts, (uint32_t)ptr_W_re, M3_R1_ss,
                                        M3_R1_tb, M3_R1_ts, (uint32_t)ptr_rr, M3_W0_ss, M3_W0_tb, M3_W0_ts);
                set_simbacore_csr(M3_OSGEMM, seqLen, M3_dPerB, M3_dPerB, 1, 1);
                start_simbacore_and_streamers(0, 0, 0, 0);
                wait_simbacore_and_streamer();
                simbacore_cycles += read_simbacore_perf_counter();

                write_csr(BASE_PTR_READER_0_LOW, (uint32_t)x_im_b);
                write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_ir);
                start_simbacore_and_streamers(0, 0, 0, 0);
                wait_simbacore_and_streamer();
                simbacore_cycles += read_simbacore_perf_counter();

                write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_W_im);
                write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_ii);
                start_simbacore_and_streamers(0, 0, 0, 0);
                wait_simbacore_and_streamer();
                simbacore_cycles += read_simbacore_perf_counter();

                write_csr(BASE_PTR_READER_0_LOW, (uint32_t)x_re_b);
                write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_ri);
                start_simbacore_and_streamers(0, 0, 0, 0);
                wait_simbacore_and_streamer();
                simbacore_cycles += read_simbacore_perf_counter();

                // ---------------- Per-side SIMD fuse -------------------------
                // out_re = quantize( relu?( (rr - ii) + b_re ) )
                // out_im = quantize( relu?( (ri + ir) + b_im ) )
                for (int side = 0; side < 2; side++) {
                    uint8_t* pos      = (side == 0) ? ptr_rr : ptr_ri;
                    uint8_t* neg      = (side == 0) ? ptr_ii : ptr_ir;
                    uint16_t* bias_b  = (side == 0) ? b_re_bc_b : b_im_bc_b;
                    uint8_t* out_full = (side == 0) ? out_re_b : out_im_b;
                    uint32_t binop    = (side == 0) ? M9_SIMD_SUB_BF16 : M8_SIMD_ADD_BF16;

                    // Widen FP8 → BF16 (pos): rr / ri → bf16_a
                    set_simd_streamer_no_b((uint32_t)pos, M3_R7_widen_ss, M3_R7_widen_tb, M3_R7_widen_ts,
                                           (uint32_t)ptr_bf16_a, M3_W3_widen_ss, M3_W3_widen_tb, M3_W3_widen_ts);
                    set_simbacore_csr(M25_SIMD_NOOP_FP8_REQUANT, seqLen, M3_dPerB, M3_dPerB, 1, 1);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();

                    // Widen FP8 → BF16 (neg): ii / ir → bf16_b
                    write_csr(BASE_PTR_READER_7_LOW, (uint32_t)neg);
                    write_csr(BASE_PTR_WRITER_3_LOW, (uint32_t)ptr_bf16_b);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();

                    // BF16 binop: bf16_a (±) bf16_b → bf16_a
                    set_simd_streamer_csr((uint32_t)ptr_bf16_a, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts,
                                          (uint32_t)ptr_bf16_b, M3_R13_bf16_ss, M3_R13_bf16_tb, M3_R13_bf16_ts,
                                          (uint32_t)ptr_bf16_a, M3_W3_bf16_ss, M3_W3_bf16_tb, M3_W3_bf16_ts);
                    set_simbacore_csr(binop, seqLen, M3_dPerB, M3_dPerB, 1, 1);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();

                    // BF16 ADD bias broadcast: bf16_a + bias_bcast → bf16_a
                    // Layer 1: doRelu=1 fused into mode (M3_SIMD_ADD_BF16_RELU).
                    // Layer 2: plain ADD (no relu).
                    write_csr(BASE_PTR_READER_13_LOW, (uint32_t)bias_b);
                    write_csr(MODE, add_bias_mode);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();

                    // Narrow BF16 → FP8: bf16_a → ConvFormat output (per-branch).
                    set_simd_streamer_no_b((uint32_t)ptr_bf16_a, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts,
                                           (uint32_t)out_full, M3_W3_fp8_ss, M3_W3_fp8_tb, M3_W3_fp8_ts);
                    set_simbacore_csr(M24_SIMD_NOOP_BF16_REQUANT, seqLen, M3_dPerB, M3_dPerB, 1, 1);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();
                }
            }
            snrt_cluster_hw_barrier();
        }

        if (snrt_global_core_idx() == 0) printf("[%u cc] Layer %d done\n", snrt_mcycle(), layer);
    }

    // ---- Verify (against ConvFormat-per-branch references) -------------------
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] Simbacore total (sum over launches): %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%u cc] Snitch elapsed: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err +=
            check_result_sample(ptr_l1_re, M3_output_1_real, M3_test_samples_output_1_real, nb_test_samples, "l1_real");
        err +=
            check_result_sample(ptr_l1_im, M3_output_1_imag, M3_test_samples_output_1_imag, nb_test_samples, "l1_imag");
        err +=
            check_result_sample(ptr_l2_re, M3_output_2_real, M3_test_samples_output_2_real, nb_test_samples, "l2_real");
        err +=
            check_result_sample(ptr_l2_im, M3_output_2_imag, M3_test_samples_output_2_imag, nb_test_samples, "l2_imag");
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 4 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}
