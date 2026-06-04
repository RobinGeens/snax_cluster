// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Dual-core EinFFT MLP: the four per-side matmuls run as two IS_OSGEMM calls,
// keeping the OS-core (real side: rr, ii -> FP8 ConvFormat) and IS-core
// (imag side: ri, ir -> BF16 flattenCD) busy in parallel. The real fuse is
// einfft's ConvFormat chain; the imag fuse is a shorter BF16 chain (no widen).
// No N-tiling (the IS_OSGEMM shared CSRs couple OS-N and IS-K).
// See docs/dataflow/06_einfft_mlp.md.

#include "data.h"
#include "snax-simbacore-lib.h"

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

// One combined OS+IS GEMM launch. Both cores compute their own matmul in
// parallel; bases are rebound by the caller between launches.
static inline void run_iosgemm(void) {
    write_csr(MODE, M3_IS_OSGEMM_NO_REQUANT);
    start_simbacore_and_streamers(0, 0, M3_R11_en, 0);
    write_csr(STREAMER_START_CSR, 0);
    write_csr(SIMBACORE_START, 0);
    write_csr(DELAYED_START_READER_10, 0);
    write_csr(DELAYED_START_READER_11, 0);
    while (read_csr(SIMBACORE_BUSY));
    while (read_csr(STREAMER_BUSY_CSR));
}

static inline void simd_pulse(void) {
    _set_streamer_start();
    _set_simbacore_start();
    write_csr(STREAMER_START_CSR, 0);
    write_csr(SIMBACORE_START, 0);
}

int main(void) {
    int err = 0;

    static uint8_t* l3_out_re[2] = {NULL, NULL};
    static uint8_t* l3_out_im[2] = {NULL, NULL};
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        l3_out_re[0] = (uint8_t*)snrt_l3alloc(M3_length_output_1_real);
        l3_out_im[0] = (uint8_t*)snrt_l3alloc(M3_length_output_1_imag);
        l3_out_re[1] = (uint8_t*)snrt_l3alloc(M3_length_output_2_real);
        l3_out_im[1] = (uint8_t*)snrt_l3alloc(M3_length_output_2_imag);
    }
    snrt_cluster_hw_barrier();

    void* tcdm = snrt_l1_next();

    // ---- TCDM layout (all FULL per branch; L3-staged per (layer, branch)) ----
    uint8_t* p = (uint8_t*)tcdm;
    uint8_t* ptr_x_re    = p;             p = _ALIGN64(p + M3_length_x_branch);   // OS A (flatA)
    uint8_t* ptr_x_im    = p;             p = _ALIGN64(p + M3_length_x_branch);
    uint8_t* ptr_x_re_cv = p;             p = _ALIGN64(p + M3_length_x_branch);   // IS A (ConvFormat)
    uint8_t* ptr_x_im_cv = p;             p = _ALIGN64(p + M3_length_x_branch);
    uint8_t* ptr_w_re    = p;             p = _ALIGN64(p + M3_length_w_branch);   // OS weights
    uint8_t* ptr_w_im    = p;             p = _ALIGN64(p + M3_length_w_branch);
    uint8_t* ptr_w_re_is = p;             p = _ALIGN64(p + M3_length_w_branch);   // IS weights
    uint8_t* ptr_w_im_is = p;             p = _ALIGN64(p + M3_length_w_branch);
    uint8_t* ptr_rr      = p;             p = _ALIGN64(p + M3_length_d);          // OS scratch (ConvFormat FP8)
    uint8_t* ptr_ii      = p;             p = _ALIGN64(p + M3_length_d);
    // IS psum: seeded with b_im, then accumulates ri (call A) and ir (call B). BF16.
    uint8_t* ptr_P       = p;             p = _ALIGN64(p + M3_length_cd);
    uint16_t* ptr_bf16_a = (uint16_t*)p;  p = _ALIGN64(p + M3_length_bf16);       // real SIMD staging
    uint16_t* ptr_bf16_b = (uint16_t*)p;  p = _ALIGN64(p + M3_length_bf16);
    uint16_t* ptr_b_re   = (uint16_t*)p;  p = _ALIGN64(p + M3_length_bias_mini_branch);  // real bias (conv-walk)
    uint8_t* ptr_out_re  = p;             p = _ALIGN64(p + M3_length_out_branch);
    uint8_t* ptr_out_im  = p;             p = _ALIGN64(p + M3_length_out_branch);

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    uint32_t simbacore_cycles = 0, start_cycles = 0;
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: einfft-tiled-is-osgemm (L=%u, dModel=%u, dPerB=%u, nBranches=%u)\n\n",
               seqLen, dModel, M3_dPerB, M3_nBranches);
        start_cycles = snrt_mcycle();
    }

    for (int layer = 0; layer < 2; layer++) {
        uint8_t* x_re_l3    = (layer == 0) ? M3_x_real      : M3_x_2_real;
        uint8_t* x_im_l3    = (layer == 0) ? M3_x_imag      : M3_x_2_imag;
        uint8_t* x_re_cv_l3 = (layer == 0) ? M3_x_real_conv : M3_x_2_real_conv;
        uint8_t* x_im_cv_l3 = (layer == 0) ? M3_x_imag_conv : M3_x_2_imag_conv;
        uint8_t* w_re_l3    = (layer == 0) ? M3_weight_1_real    : M3_weight_2_real;
        uint8_t* w_im_l3    = (layer == 0) ? M3_weight_1_imag    : M3_weight_2_imag;
        uint8_t* w_re_is_l3 = (layer == 0) ? M3_weight_1_real_is : M3_weight_2_real_is;
        uint8_t* w_im_is_l3 = (layer == 0) ? M3_weight_1_imag_is : M3_weight_2_imag_is;
        uint16_t* b_re_l3   = (layer == 0) ? M3_bias_1_real_mini : M3_bias_2_real_mini;
        uint16_t* b_im_l3   = (layer == 0) ? M3_bias_1_imag_full : M3_bias_2_imag_full;
        uint8_t* out_re_l3  = l3_out_re[layer];
        uint8_t* out_im_l3  = l3_out_im[layer];
        uint32_t add_bias_mode  = (layer == 0) ? M3_SIMD_ADD_BF16_RELU : M8_SIMD_ADD_BF16;          // real bias-add
        uint32_t narrow_im_mode = (layer == 0) ? M3_SIMD_NOOP_BF16_REQUANT_RELU : M24_SIMD_NOOP_BF16_REQUANT;

        for (uint32_t b = 0; b < M3_nBranches; b++) {
            // ---- Stage 1: DMA this branch's inputs; seed the IS psum P with b_im ----
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_x_re,    x_re_l3    + b * M3_length_x_branch, M3_length_x_branch);
                snrt_dma_start_1d(ptr_x_im,    x_im_l3    + b * M3_length_x_branch, M3_length_x_branch);
                snrt_dma_start_1d(ptr_x_re_cv, x_re_cv_l3 + b * M3_length_x_branch, M3_length_x_branch);
                snrt_dma_start_1d(ptr_x_im_cv, x_im_cv_l3 + b * M3_length_x_branch, M3_length_x_branch);
                snrt_dma_start_1d(ptr_w_re,    w_re_l3    + b * M3_length_w_branch, M3_length_w_branch);
                snrt_dma_start_1d(ptr_w_im,    w_im_l3    + b * M3_length_w_branch, M3_length_w_branch);
                snrt_dma_start_1d(ptr_w_re_is, w_re_is_l3 + b * M3_length_w_branch, M3_length_w_branch);
                snrt_dma_start_1d(ptr_w_im_is, w_im_is_l3 + b * M3_length_w_branch, M3_length_w_branch);
                snrt_dma_start_1d((uint8_t*)ptr_b_re, (uint8_t*)(b_re_l3) + b * M3_length_bias_mini_branch,
                                  M3_length_bias_mini_branch);
                // Seed psum P = b_im (IS-core C-input); call A/B accumulate ri/ir on top.
                snrt_dma_start_1d(ptr_P, (uint8_t*)(b_im_l3) + b * M3_length_bias_im_branch, M3_length_cd);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();

            // ---- Stage 2: compute (core 0) ----
            if (snrt_global_core_idx() == 0) {
                // GEMM dims: OS K=dPerB, OS N = IS K = dPerB, IS N(dFinal)=dPerB.
                set_simbacore_csr(M3_IS_OSGEMM_NO_REQUANT, seqLen, M3_dPerB, M3_dPerB, 1, M3_dPerB);

                // Streamer setup for the combined GEMM (R0/R1/W0 OS, R11/R12/R13/W3 IS).
                // Call A bases: OS rr = x_re @ W_re ; IS ri = x_re_cv @ W_im_is.
                set_streamer_csr(
                    (uint32_t)ptr_x_re,    M3_R0_ss, M3_R0_tb, M3_R0_ts, M3_R0_en,     // R0
                    (uint32_t)ptr_w_re,    M3_R1_ss, M3_R1_tb, M3_R1_ts, M3_R1_en,     // R1
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,        // R2..R5
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,        // R6..R9
                    0, 0, 0, 0, 0,                                                     // R10
                    (uint32_t)ptr_x_re_cv, M3_R11_ss, M3_R11_tb, M3_R11_ts, M3_R11_en, // R11
                    (uint32_t)ptr_w_im_is, M3_R12_ss, M3_R12_tb, M3_R12_ts, M3_R12_en, // R12
                    (uint32_t)ptr_P,       M3_R13_ss, M3_R13_tb, M3_R13_ts, M3_R13_en, // R13: psum-in (= b_im)
                    (uint32_t)ptr_rr,      M3_W0_ss, M3_W0_tb, M3_W0_ts, M3_W0_en,     // W0
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                                      // W1..W2
                    (uint32_t)ptr_P,       M3_W3_ss, M3_W3_tb, M3_W3_ts, M3_W3_en      // W3: psum-out
                );
                run_iosgemm();  // P = b_im + ri (= x_re @ W_im)
                simbacore_cycles += read_simbacore_perf_counter();

                // Call B bases: OS ii = x_im @ W_im ; IS ir = x_im_cv @ W_re_is, accumulating into P.
                write_csr(BASE_PTR_READER_0_LOW, (uint32_t)ptr_x_im);
                write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_w_im);
                write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_ii);
                write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_x_im_cv);
                write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_w_re_is);
                // R13/W3 stay on P (psum read-modify-write).
                run_iosgemm();  // P = b_im + ri + ir
                simbacore_cycles += read_simbacore_perf_counter();

                // ============ REAL fuse: out_re = ReLU?(rr - ii + b_re) ============
                // ConvFormat FP8 chain (widen, SUB, bias-add, narrow) — same as einfft.
                set_simbacore_csr(M25_SIMD_NOOP_FP8_REQUANT, seqLen, M3_dPerB, M3_dPerB, 1, 1);
                // 1: widen rr -> bf16_a
                set_simd_streamer_no_b((uint32_t)ptr_rr, M3_R7_widen_ss, M3_R7_widen_tb, M3_R7_widen_ts,
                                       (uint32_t)ptr_bf16_a, M3_W3_widen_ss, M3_W3_widen_tb, M3_W3_widen_ts);
                write_csr(MODE, M25_SIMD_NOOP_FP8_REQUANT);
                simd_pulse();
                write_csr(BASE_PTR_READER_7_LOW, (uint32_t)ptr_ii);
                write_csr(BASE_PTR_WRITER_3_LOW, (uint32_t)ptr_bf16_b);
                while (read_csr(SIMBACORE_BUSY)); while (read_csr(STREAMER_BUSY_CSR));
                simbacore_cycles += read_simbacore_perf_counter();
                // 2: widen ii -> bf16_b ; prep SUB streamers
                simd_pulse();
                set_simd_streamer_csr((uint32_t)ptr_bf16_a, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts,
                                      (uint32_t)ptr_bf16_b, M3_R13_bf16_ss, M3_R13_bf16_tb, M3_R13_bf16_ts,
                                      (uint32_t)ptr_bf16_a, M3_W3_bf16_ss, M3_W3_bf16_tb, M3_W3_bf16_ts);
                while (read_csr(SIMBACORE_BUSY)); while (read_csr(STREAMER_BUSY_CSR));
                simbacore_cycles += read_simbacore_perf_counter();
                // 3: SUB bf16_a - bf16_b -> bf16_a
                write_csr(MODE, M9_SIMD_SUB_BF16);
                simd_pulse();
                while (read_csr(SIMBACORE_BUSY)); while (read_csr(STREAMER_BUSY_CSR));
                simbacore_cycles += read_simbacore_perf_counter();
                // 4: ADD bias (conv-walk R13) -> bf16_a ; prep narrow
                write_csr(BASE_PTR_READER_13_LOW, (uint32_t)ptr_b_re);
                write_csr(T_BOUND_BASE_READER_13 + 0, M3_R13_bias_tb[0]);
                write_csr(T_BOUND_BASE_READER_13 + 1, M3_R13_bias_tb[1]);
                write_csr(T_BOUND_BASE_READER_13 + 2, M3_R13_bias_tb[2]);
                write_csr(T_BOUND_BASE_READER_13 + 3, M3_R13_bias_tb[3]);
                write_csr(T_STRIDE_BASE_READER_13 + 0, M3_R13_bias_ts[0]);
                write_csr(T_STRIDE_BASE_READER_13 + 1, M3_R13_bias_ts[1]);
                write_csr(T_STRIDE_BASE_READER_13 + 2, M3_R13_bias_ts[2]);
                write_csr(T_STRIDE_BASE_READER_13 + 3, M3_R13_bias_ts[3]);
                write_csr(MODE, add_bias_mode);
                simd_pulse();
                set_simd_streamer_no_b((uint32_t)ptr_bf16_a, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts,
                                       (uint32_t)ptr_out_re, M3_W3_fp8_ss, M3_W3_fp8_tb, M3_W3_fp8_ts);
                while (read_csr(SIMBACORE_BUSY)); while (read_csr(STREAMER_BUSY_CSR));
                simbacore_cycles += read_simbacore_perf_counter();
                // 5: narrow bf16_a -> out_re (FP8)
                write_csr(MODE, M24_SIMD_NOOP_BF16_REQUANT);
                simd_pulse();
                while (read_csr(SIMBACORE_BUSY)); while (read_csr(STREAMER_BUSY_CSR));
                simbacore_cycles += read_simbacore_perf_counter();

                // ============ IMAG fuse: out_im = ReLU?(P) narrowed to FP8 ============
                // The add (ri+ir) and bias were folded into the IS-core psum P. Only the
                // ReLU + BF16->FP8 narrow remains: one SIMD launch.
                set_simbacore_csr(M24_SIMD_NOOP_BF16_REQUANT, seqLen, M3_dPerB, M3_dPerB, 1, 1);
                set_simd_streamer_no_b((uint32_t)ptr_P, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts,
                                       (uint32_t)ptr_out_im, M3_W3_fp8_ss, M3_W3_fp8_tb, M3_W3_fp8_ts);
                write_csr(MODE, narrow_im_mode);
                simd_pulse();
                while (read_csr(SIMBACORE_BUSY)); while (read_csr(STREAMER_BUSY_CSR));
                simbacore_cycles += read_simbacore_perf_counter();
            }
            snrt_cluster_hw_barrier();

            // ---- Stage 3: spill outputs to L3 ----
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(out_re_l3 + b * M3_length_out_branch, ptr_out_re, M3_length_out_branch);
                snrt_dma_start_1d(out_im_l3 + b * M3_length_out_branch, ptr_out_im, M3_length_out_branch);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();
        }
    }

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%u cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(l3_out_re[0], M3_output_1_real, M3_test_samples_output_1_real, nb_test_samples,
                                   "l1_real");
        err += check_result_sample(l3_out_im[0], M3_output_1_imag, M3_test_samples_output_1_imag, nb_test_samples,
                                   "l1_imag");
        err += check_result_sample(l3_out_re[1], M3_output_2_real, M3_test_samples_output_2_real, nb_test_samples,
                                   "l2_real");
        err += check_result_sample(l3_out_im[1], M3_output_2_imag, M3_test_samples_output_2_imag, nb_test_samples,
                                   "l2_imag");
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 4 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}
