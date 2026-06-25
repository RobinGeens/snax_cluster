// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// bank-conflict-free einfft-tiled-is-osgemm. Every buffer the two parallel
// GEMM cores touch uses the bank-partitioned skip-128 layout: the OS-core lives in TCDM banks 0-15 and the IS-core
// in banks 16-31
//
// See docs/dataflow/20_double_gemm_conflict_free.md and and docs/dataflow/06_einfft_mlp.md.

#include "data.h"
#include "snax-simbacore-lib.h"

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

#define SB 128u  // skip-128 block: low 128 B of every 256 B = banks 0-15, high 128 B = banks 16-31
// Physical footprint of a logical buffer of `len` bytes in the skip-128 layout (len % 128 == 0).
#define SKIP_BYTES(len) (2u * (uint32_t)(len))

#define NB_BRANCHES M3_nBranches
#define NB_ITERS (2 * NB_BRANCHES)  // 2 layers x nBranches, layer-major
#define NB_STAGES 3

// Allocate a [2] ping-pong pair of skip-128 buffers off the running pointer `p`.
#define ALLOC_SKIP2(arr, len) \
    do {                      \
        arr[0] = p;           \
        p += SKIP_BYTES(len); \
        arr[1] = p;           \
        p += SKIP_BYTES(len); \
    } while (0)
// Allocate a [2] ping-pong pair of contiguous buffers off the running pointer `s`.
#define ALLOC_C2(arr, len)            \
    do {                              \
        arr[0] = s;                   \
        s      = _ALIGN64(s + (len)); \
        arr[1] = s;                   \
        s      = _ALIGN64(s + (len)); \
    } while (0)

// DMA a contiguous L3 buffer into a skip-128 TCDM buffer (each 128 B block on a 256 B centre).
static inline void dma_scatter_128(void* dst_skip, const void* src_contig, uint32_t len) {
    snrt_dma_start_2d(dst_skip, src_contig, SB, 2 * SB, SB, len / SB);
}

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

    // Per-iteration L3 sources/sinks, indexed [layer]. The DM core offsets each by branch.
    uint8_t* x_re_l3[2]    = {M3_x_real, M3_x_2_real};
    uint8_t* x_im_l3[2]    = {M3_x_imag, M3_x_2_imag};
    uint8_t* x_re_cv_l3[2] = {M3_x_real_conv, M3_x_2_real_conv};
    uint8_t* x_im_cv_l3[2] = {M3_x_imag_conv, M3_x_2_imag_conv};
    uint8_t* w_re_l3[2]    = {M3_weight_1_real, M3_weight_2_real};
    uint8_t* w_im_l3[2]    = {M3_weight_1_imag, M3_weight_2_imag};
    uint8_t* w_re_is_l3[2] = {M3_weight_1_real_is, M3_weight_2_real_is};
    uint8_t* w_im_is_l3[2] = {M3_weight_1_imag_is, M3_weight_2_imag_is};
    uint16_t* b_re_l3[2]   = {M3_bias_1_real_mini, M3_bias_2_real_mini};
    uint16_t* b_im_l3[2]   = {M3_bias_1_imag_full, M3_bias_2_imag_full};

    // ---- TCDM layout: two OVERLAPPED skip-128 arenas + a contiguous SIMD region ----
    // OS half in banks 0-15 (base 256-aligned -> addr[7]=0); IS half in banks 16-31
    // (same base + 128 -> addr[7]=1). Each skip buffer occupies 2x its logical size but
    // uses only the half of every 256 B block its core owns; the other half is exactly
    // what the other core uses. So the two heaps share the SAME base and overlap: OS fills
    // the low 128 B of each block, IS the high 128 B. They can never share a bank, yet no
    // half is wasted -> resident footprint is 2*max(OS, IS), not 2*OS + 2*IS.
    // DMA'd buffers are double-buffered ([2]); the compute-only scratch (rr/ii/bf16) is
    // single-buffered (compute is serialized).
    uint32_t os_base = ((uint32_t)snrt_l1_next() + 255u) & ~255u;
    uint8_t* p       = (uint8_t*)os_base;
    uint8_t* ptr_x_re[2];
    ALLOC_SKIP2(ptr_x_re, M3_length_x_branch);  // OS A (flatA)
    uint8_t* ptr_x_im[2];
    ALLOC_SKIP2(ptr_x_im, M3_length_x_branch);
    uint8_t* ptr_w_re[2];
    ALLOC_SKIP2(ptr_w_re, M3_length_w_branch);  // OS weights
    uint8_t* ptr_w_im[2];
    ALLOC_SKIP2(ptr_w_im, M3_length_w_branch);
    uint8_t* ptr_rr = p;
    p += SKIP_BYTES(M3_length_d);  // OS scratch (single)
    uint8_t* ptr_ii = p;
    p += SKIP_BYTES(M3_length_d);
    uint32_t os_end = (uint32_t)p;

    uint32_t is_base = os_base + SB;  // SAME base + 128 -> addr[7]=1, overlaps the OS heap
    p                = (uint8_t*)is_base;
    uint8_t* ptr_x_re_cv[2];
    ALLOC_SKIP2(ptr_x_re_cv, M3_length_x_branch);  // IS A (ConvFormat)
    uint8_t* ptr_x_im_cv[2];
    ALLOC_SKIP2(ptr_x_im_cv, M3_length_x_branch);
    uint8_t* ptr_w_re_is[2];
    ALLOC_SKIP2(ptr_w_re_is, M3_length_w_branch);  // IS weights
    uint8_t* ptr_w_im_is[2];
    ALLOC_SKIP2(ptr_w_im_is, M3_length_w_branch);
    uint8_t* ptr_P[2];
    ALLOC_SKIP2(ptr_P, M3_length_cd);  // IS psum (seeded w/ b_im)
    uint32_t is_end = (uint32_t)p;

    // Contiguous SIMD-only region (single array runs the fuse -> no cross-core contention),
    // placed past whichever overlapped heap reaches higher.
    uint8_t* s           = _ALIGN64((uint8_t*)(os_end > is_end ? os_end : is_end));
    uint16_t* ptr_bf16_a = (uint16_t*)s;
    s                    = _ALIGN64(s + M3_length_bf16);  // real SIMD staging (single)
    uint16_t* ptr_bf16_b = (uint16_t*)s;
    s                    = _ALIGN64(s + M3_length_bf16);
    uint16_t* ptr_b_re[2];
    ALLOC_C2(ptr_b_re, M3_length_bias_mini_branch);
    uint8_t* ptr_out_re[2];
    ALLOC_C2(ptr_out_re, M3_length_out_branch);
    uint8_t* ptr_out_im[2];
    ALLOC_C2(ptr_out_im, M3_length_out_branch);

    uint32_t simbacore_cycles = 0, start_cycles = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: einfft-double-conflictfree (L=%u, dModel=%u, dPerB=%u, nBranches=%u)\n\n", seqLen,
               dModel, M3_dPerB, M3_nBranches);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
    }

    // Software-pipelined: iteration i in [0, NB_ITERS + NB_STAGES - 1).
    //   transfer_in  iteration i    (DM core)   -> buf i%2
    //   compute      iteration i-1  (core 0)    -> buf (i-1)%2
    //   transfer_out iteration i-2  (DM core)   -> buf (i-2)%2
    for (uint32_t i = 0; i < NB_ITERS + NB_STAGES - 1; i++) {
        // ---- Stage 1: DMA inputs for iteration i (skip-128 scatter) + seed psum P ----
        if (i < NB_ITERS && snrt_is_dm_core()) {
            int buf = i % 2, layer = i / NB_BRANCHES, b = i % NB_BRANCHES;
            dma_scatter_128(ptr_x_re[buf], x_re_l3[layer] + b * M3_length_x_branch, M3_length_x_branch);
            dma_scatter_128(ptr_x_im[buf], x_im_l3[layer] + b * M3_length_x_branch, M3_length_x_branch);
            dma_scatter_128(ptr_x_re_cv[buf], x_re_cv_l3[layer] + b * M3_length_x_branch, M3_length_x_branch);
            dma_scatter_128(ptr_x_im_cv[buf], x_im_cv_l3[layer] + b * M3_length_x_branch, M3_length_x_branch);
            dma_scatter_128(ptr_w_re[buf], w_re_l3[layer] + b * M3_length_w_branch, M3_length_w_branch);
            dma_scatter_128(ptr_w_im[buf], w_im_l3[layer] + b * M3_length_w_branch, M3_length_w_branch);
            dma_scatter_128(ptr_w_re_is[buf], w_re_is_l3[layer] + b * M3_length_w_branch, M3_length_w_branch);
            dma_scatter_128(ptr_w_im_is[buf], w_im_is_l3[layer] + b * M3_length_w_branch, M3_length_w_branch);
            // Real bias stays contiguous (read by the single-array SIMD fuse).
            snrt_dma_start_1d((uint8_t*)ptr_b_re[buf], (uint8_t*)(b_re_l3[layer]) + b * M3_length_bias_mini_branch,
                              M3_length_bias_mini_branch);
            // Seed psum P = b_im (IS-core C-input, skip-128); calls A/B accumulate ri/ir on top.
            dma_scatter_128(ptr_P[buf], (uint8_t*)(b_im_l3[layer]) + b * M3_length_bias_im_branch, M3_length_cd);
        }

        // ---- Stage 2: compute iteration i-1 (core 0) ----
        if (i >= 1 && i < NB_ITERS + 1 && snrt_global_core_idx() == 0) {
            int buf = (i - 1) % 2, layer = (i - 1) / NB_BRANCHES;
            uint8_t* out_re         = ptr_out_re[buf];
            uint8_t* out_im         = ptr_out_im[buf];
            uint32_t add_bias_mode  = (layer == 0) ? M3_SIMD_ADD_BF16_RELU : M8_SIMD_ADD_BF16;  // real bias-add
            uint32_t narrow_im_mode = (layer == 0) ? M3_SIMD_NOOP_BF16_REQUANT_RELU : M24_SIMD_NOOP_BF16_REQUANT;

            // GEMM dims: OS K=dPerB, OS N = IS K = dPerB, IS N(dFinal)=dPerB.
            set_simbacore_csr(M3_IS_OSGEMM_NO_REQUANT, seqLen, M3_dPerB, M3_dPerB, 1, M3_dPerB);

            // Streamer setup for the combined GEMM (R0/R1/W0 OS, R11/R12/R13/W3 IS).
            // The skip-128 walk is encoded in the data.h bounds/strides; only the bases move.
            // Call A bases: OS rr = x_re @ W_re ; IS ri = x_re_cv @ W_im_is.
            set_streamer_csr((uint32_t)ptr_x_re[buf], M3_R0_ss, M3_R0_tb, M3_R0_ts, M3_R0_en,         // R0
                             (uint32_t)ptr_w_re[buf], M3_R1_ss, M3_R1_tb, M3_R1_ts, M3_R1_en,         // R1
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,              // R2..R5
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,              // R6..R9
                             0, 0, 0, 0, 0,                                                           // R10
                             (uint32_t)ptr_x_re_cv[buf], M3_R11_ss, M3_R11_tb, M3_R11_ts, M3_R11_en,  // R11
                             (uint32_t)ptr_w_im_is[buf], M3_R12_ss, M3_R12_tb, M3_R12_ts, M3_R12_en,  // R12
                             (uint32_t)ptr_P[buf], M3_R13_ss, M3_R13_tb, M3_R13_ts, M3_R13_en,  // R13: psum-in (b_im)
                             (uint32_t)ptr_rr, M3_W0_ss, M3_W0_tb, M3_W0_ts, M3_W0_en,          // W0
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                                      // W1..W2
                             (uint32_t)ptr_P[buf], M3_W3_ss, M3_W3_tb, M3_W3_ts, M3_W3_en       // W3: psum-out
            );
            run_iosgemm();  // P = b_im + ri (= x_re @ W_im)
            simbacore_cycles += read_simbacore_perf_counter();

            // Call B bases: OS ii = x_im @ W_im ; IS ir = x_im_cv @ W_re_is, accumulating into P.
            write_csr(BASE_PTR_READER_0_LOW, (uint32_t)ptr_x_im[buf]);
            write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_w_im[buf]);
            write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_ii);
            write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_x_im_cv[buf]);
            write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_w_re_is[buf]);
            // R13/W3 stay on P (psum read-modify-write).
            run_iosgemm();  // P = b_im + ri + ir
            simbacore_cycles += read_simbacore_perf_counter();

            // ============ REAL fuse: out_re = ReLU?(rr - ii + b_re) ============
            // ConvFormat FP8 chain (widen, SUB, bias-add, narrow) — same as einfft.
            // rr/ii live skip-128 in the OS half, so the widen reader walks skip-128.
            set_simbacore_csr(M25_SIMD_NOOP_FP8_REQUANT, seqLen, M3_dPerB, M3_dPerB, 1, 1);
            // 1: widen rr -> bf16_a
            set_simd_streamer_no_b((uint32_t)ptr_rr, M3_R7_widen_ss, M3_R7_widen_tb, M3_R7_widen_ts,
                                   (uint32_t)ptr_bf16_a, M3_W3_widen_ss, M3_W3_widen_tb, M3_W3_widen_ts);
            write_csr(MODE, M25_SIMD_NOOP_FP8_REQUANT);
            simd_pulse();
            write_csr(BASE_PTR_READER_7_LOW, (uint32_t)ptr_ii);
            write_csr(BASE_PTR_WRITER_3_LOW, (uint32_t)ptr_bf16_b);
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles += read_simbacore_perf_counter();
            // 2: widen ii -> bf16_b ; prep SUB streamers (bf16_a/bf16_b contiguous)
            simd_pulse();
            set_simd_streamer_csr((uint32_t)ptr_bf16_a, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts,
                                  (uint32_t)ptr_bf16_b, M3_R13_bf16_ss, M3_R13_bf16_tb, M3_R13_bf16_ts,
                                  (uint32_t)ptr_bf16_a, M3_W3_bf16_ss, M3_W3_bf16_tb, M3_W3_bf16_ts);
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles += read_simbacore_perf_counter();
            // 3: SUB bf16_a - bf16_b -> bf16_a
            write_csr(MODE, M9_SIMD_SUB_BF16);
            simd_pulse();
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles += read_simbacore_perf_counter();
            // 4: ADD bias (conv-walk R13) -> bf16_a ; prep narrow
            write_csr(BASE_PTR_READER_13_LOW, (uint32_t)ptr_b_re[buf]);
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
            set_simd_streamer_no_b((uint32_t)ptr_bf16_a, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts, (uint32_t)out_re,
                                   M3_W3_fp8_ss, M3_W3_fp8_tb, M3_W3_fp8_ts);
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles += read_simbacore_perf_counter();
            // 5: narrow bf16_a -> out_re (FP8)
            write_csr(MODE, M24_SIMD_NOOP_BF16_REQUANT);
            simd_pulse();
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles += read_simbacore_perf_counter();

            // ============ IMAG fuse: out_im = ReLU?(P) narrowed to FP8 ============
            // The add (ri+ir) and bias were folded into the IS-core psum P. Only the
            // ReLU + BF16->FP8 narrow remains: one SIMD launch. P lives skip-128 in the
            // IS half, so the reader walks skip-128 (R7_bf16_skip); out_im is contiguous.
            set_simbacore_csr(M24_SIMD_NOOP_BF16_REQUANT, seqLen, M3_dPerB, M3_dPerB, 1, 1);
            set_simd_streamer_no_b((uint32_t)ptr_P[buf], M3_R7_bf16_skip_ss, M3_R7_bf16_skip_tb, M3_R7_bf16_skip_ts,
                                   (uint32_t)out_im, M3_W3_fp8_ss, M3_W3_fp8_tb, M3_W3_fp8_ts);
            write_csr(MODE, narrow_im_mode);
            simd_pulse();
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles += read_simbacore_perf_counter();
        }

        // ---- Stage 3: spill iteration i-2 outputs to L3 (contiguous) ----
        if (i >= NB_STAGES - 1 && snrt_is_dm_core()) {
            uint32_t it = i - (NB_STAGES - 1);
            int buf = it % 2, layer = it / NB_BRANCHES, b = it % NB_BRANCHES;
            snrt_dma_start_1d(l3_out_re[layer] + b * M3_length_out_branch, ptr_out_re[buf], M3_length_out_branch);
            snrt_dma_start_1d(l3_out_im[layer] + b * M3_length_out_branch, ptr_out_im[buf], M3_length_out_branch);
        }

        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
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
