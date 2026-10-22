// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled EinFFT with per-tile fuse.
//   - The N (= D/4 output-channel) axis is tiled. The four per-tile matmuls are run as
//     two concat OSGEMMs that share an activation (x_re @ [W_re|W_im] = [rr|ri], and
//     x_im @ [W_re|W_im] = [ir|ii]), then the SIMD fuse runs for that tile. The OS-core
//     scratch (out_A/out_B), BF16 staging, and output buffers only ever hold one tile,
//     so large (L, dModel) fit in TCDM. The concat keeps every OSGEMM at N>=2 (the HW
//     requirement) with no wasted compute, even at dModel=96 where each matmul is N=1.
//   - Weight tiles are prefetched one tile ahead (ping-pong); W_re and W_im are packed
//     adjacently so the concat OSGEMM reads them as one weight.
//   - Output tiles are spilled to L3 one tile behind (depth-2 ping-pong),
//     overlapping the spill DMA with the next tile's compute.

#include "data.h"
#include "snax-simbacore-lib.h"

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

// Restore R13 to a safe disabled state: the bias walk leaves zero temporal strides,
// and the stride-zero bug would turn the next launch's bound0=0 disable into bound=1.
static inline void r13_safe_disable(void) {
    write_csr(S_STRIDE_BASE_READER_13, 1);
    write_csr(T_BOUND_BASE_READER_13 + 0, 0);
    write_csr(T_BOUND_BASE_READER_13 + 1, 0);
    write_csr(T_BOUND_BASE_READER_13 + 2, 0);
    write_csr(T_BOUND_BASE_READER_13 + 3, 0);
    write_csr(T_STRIDE_BASE_READER_13 + 0, 1);
    write_csr(T_STRIDE_BASE_READER_13 + 1, 1);
    write_csr(T_STRIDE_BASE_READER_13 + 2, 1);
    write_csr(T_STRIDE_BASE_READER_13 + 3, 1);
}

int main(void) {
    int err = 0;

    static uint8_t* l3_out_l1_re = NULL;
    static uint8_t* l3_out_l1_im = NULL;
    static uint8_t* l3_out_l2_re = NULL;
    static uint8_t* l3_out_l2_im = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        l3_out_l1_re = (uint8_t*)snrt_l3alloc(M3_length_output_1_real);
        l3_out_l1_im = (uint8_t*)snrt_l3alloc(M3_length_output_1_imag);
        l3_out_l2_re = (uint8_t*)snrt_l3alloc(M3_length_output_2_real);
        l3_out_l2_im = (uint8_t*)snrt_l3alloc(M3_length_output_2_imag);
    }
    snrt_cluster_hw_barrier();

    void* tcdm_base_ptr = snrt_l1_next();

    // ---- TCDM allocation ------------------------------------------------
    // x stays FULL per branch (reused by every N-tile).
    uint8_t* ptr_x_re_b = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_x_im_b = _ALIGN64(ptr_x_re_b + M3_length_x_branch);

    // Mini bias: FULL per branch, double-buffered.
    uint16_t* ptr_b_re_pp[2];
    uint16_t* ptr_b_im_pp[2];
    ptr_b_re_pp[0] = (uint16_t*)_ALIGN64(ptr_x_im_b + M3_length_x_branch);
    ptr_b_re_pp[1] = (uint16_t*)_ALIGN64((uint8_t*)ptr_b_re_pp[0] + M3_length_bias_mini_branch);
    ptr_b_im_pp[0] = (uint16_t*)_ALIGN64((uint8_t*)ptr_b_re_pp[1] + M3_length_bias_mini_branch);
    ptr_b_im_pp[1] = (uint16_t*)_ALIGN64((uint8_t*)ptr_b_im_pp[0] + M3_length_bias_mini_branch);

    // Output: TILE-sized, double-buffered (spill overlap).
    uint8_t* ptr_out_re_pp[2];
    uint8_t* ptr_out_im_pp[2];
    ptr_out_re_pp[0] = _ALIGN64((uint8_t*)ptr_b_im_pp[1] + M3_length_bias_mini_branch);
    ptr_out_im_pp[0] = _ALIGN64(ptr_out_re_pp[0] + M3_length_d_tile);
    ptr_out_re_pp[1] = _ALIGN64(ptr_out_im_pp[0] + M3_length_d_tile);
    ptr_out_im_pp[1] = _ALIGN64(ptr_out_re_pp[1] + M3_length_d_tile);

    uint8_t* p_after = _ALIGN64(ptr_out_im_pp[1] + M3_length_d_tile);

    // Combined weight ping-pong: each buffer holds [W_re_tile | W_im_tile] adjacently, so the
    // OSGEMM reads them as one N_cat = 2*N_t weight (flattenB is N-tile-major). The DMA fills
    // the W_re half at +0 and the W_im half at +M3_length_w_branch_tile.
    uint8_t* ptr_W_pp[2] = {p_after, _ALIGN64(p_after + M3_length_w_cat_tile)};
    p_after              = _ALIGN64(ptr_W_pp[1] + M3_length_w_cat_tile);

    // OS-core scratches: each concat OSGEMM writes two real sub-tiles (no wasted compute).
    //   out_A = [rr | ri]  (= x_re @ [W_re|W_im])    out_B = [ir | ii]  (= x_im @ [W_re|W_im])
    uint8_t* ptr_outA    = p_after;
    p_after              = _ALIGN64(p_after + M3_length_d_cat);
    uint8_t* ptr_outB    = p_after;
    p_after              = _ALIGN64(p_after + M3_length_d_cat);
    uint16_t* ptr_bf16_a = (uint16_t*)p_after;

    uint32_t simbacore_cycles = 0, start_cycles = 0;

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: einfft-tiled (per-tile fuse) "
            "(L=%u, dModel=%u, dPerB=%u, dPerB_tile=%u, nb_tiles=%u, nBranches=%u)\n\n",
            seqLen, dModel, M3_dPerB, M3_dPerB_tile, nb_tiles, M3_nBranches);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
    }

    for (int layer = 0; layer < 2; layer++) {
        uint8_t* layer_x_re_l3   = (layer == 0) ? M3_x_real : M3_x_2_real;
        uint8_t* layer_x_im_l3   = (layer == 0) ? M3_x_imag : M3_x_2_imag;
        uint8_t* layer_w_re_l3   = (layer == 0) ? M3_weight_1_real : M3_weight_2_real;
        uint8_t* layer_w_im_l3   = (layer == 0) ? M3_weight_1_imag : M3_weight_2_imag;
        uint8_t* layer_b_re_l3   = (layer == 0) ? (uint8_t*)M3_bias_1_real_mini : (uint8_t*)M3_bias_2_real_mini;
        uint8_t* layer_b_im_l3   = (layer == 0) ? (uint8_t*)M3_bias_1_imag_mini : (uint8_t*)M3_bias_2_imag_mini;
        uint8_t* layer_out_re_l3 = (layer == 0) ? l3_out_l1_re : l3_out_l2_re;
        uint8_t* layer_out_im_l3 = (layer == 0) ? l3_out_l1_im : l3_out_l2_im;
        uint32_t add_bias_mode   = (layer == 0) ? M3_SIMD_ADD_BF16_RELU_REQUANT : M3_SIMD_ADD_BF16_REQUANT;

        for (uint32_t b = 0; b < M3_nBranches; b++) {
            int bbuf           = b & 1;
            uint8_t* w_re_b_l3 = layer_w_re_l3 + b * M3_length_w_branch;
            uint8_t* w_im_b_l3 = layer_w_im_l3 + b * M3_length_w_branch;

            // Pipelined tile loop: iter i DMAs weight tile i; compute core
            // processes tile (i-1); DM spills output of tile (i-2).
            for (uint32_t i = 0; i < nb_tiles + 1; i++) {
                int buf = i & 1;

                if (snrt_is_dm_core()) {
                    if (i < nb_tiles) {
                        // Pack [W_re_tile | W_im_tile] into the combined buffer for the concat OSGEMM.
                        snrt_dma_start_1d(ptr_W_pp[buf], w_re_b_l3 + i * M3_length_w_branch_tile,
                                          M3_length_w_branch_tile);
                        snrt_dma_start_1d(ptr_W_pp[buf] + M3_length_w_branch_tile,
                                          w_im_b_l3 + i * M3_length_w_branch_tile, M3_length_w_branch_tile);
                    }

                    // Branch input x + bias (loaded once at this branch's first iter).
                    if (i == 0) {
                        snrt_dma_start_1d(ptr_x_re_b, layer_x_re_l3 + b * M3_length_x_branch, M3_length_x_branch);
                        snrt_dma_start_1d(ptr_x_im_b, layer_x_im_l3 + b * M3_length_x_branch, M3_length_x_branch);
                        snrt_dma_start_1d((uint8_t*)ptr_b_re_pp[bbuf], layer_b_re_l3 + b * M3_length_bias_mini_branch,
                                          M3_length_bias_mini_branch);
                        snrt_dma_start_1d((uint8_t*)ptr_b_im_pp[bbuf], layer_b_im_l3 + b * M3_length_bias_mini_branch,
                                          M3_length_bias_mini_branch);
                    }

                    // Spill output tile (i-2), produced in the previous iter.
                    if (i >= 2) {
                        uint32_t st = i - 2;
                        int sbuf    = st & 1;
                        snrt_dma_start_1d(layer_out_re_l3 + b * M3_length_out_branch + st * M3_length_d_tile,
                                          ptr_out_re_pp[sbuf], M3_length_d_tile);
                        snrt_dma_start_1d(layer_out_im_l3 + b * M3_length_out_branch + st * M3_length_d_tile,
                                          ptr_out_im_pp[sbuf], M3_length_d_tile);
                    }
                    snrt_dma_wait_all();
                }

                if (i >= 1 && snrt_global_core_idx() == 0) {
                    uint32_t tile = i - 1;
                    int cbuf      = tile & 1;
                    int obuf      = tile & 1;

                    // ---- Two concat OSGEMMs for this tile (each N_cat=2*N_t real tiles) ----
                    //   out_A = x_re @ [W_re|W_im] = [rr | ri] ; out_B = x_im @ [W_re|W_im] = [ir | ii]
                    set_osgemm_streamer_csr((uint32_t)ptr_x_re_b, M3_R0_ss, M3_R0_tb, M3_R0_ts,
                                            (uint32_t)ptr_W_pp[cbuf], M3_R1_ss, M3_R1_tb, M3_R1_ts, (uint32_t)ptr_outA,
                                            M3_W0_ss, M3_W0_tb, M3_W0_ts);
                    set_simbacore_csr(M3_OSGEMM, seqLen, M3_dPerB, M3_dInner_cat, 1, 1);

                    // A: out_A = x_re @ [W_re|W_im] ; rebind for B: out_B = x_im @ [W_re|W_im]
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    write_csr(BASE_PTR_READER_0_LOW, (uint32_t)ptr_x_im_b);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_outB);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // B ; preload the full side-0 pass1' program (R7=rr, R13=ii, W3=bf16_a)
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    set_simd_streamer_csr((uint32_t)ptr_outA, M3_R7_widen_ss, M3_R7_widen_tb, M3_R7_widen_ts,
                                          (uint32_t)(ptr_outB + M3_length_d_tile), M3_R13_fp8_ss, M3_R13_fp8_tb,
                                          M3_R13_fp8_ts,  //
                                          (uint32_t)ptr_bf16_a, M3_W3_widen_ss, M3_W3_widen_tb, M3_W3_widen_ts);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // Sub-tiles inside the concat outputs (ConvFormat is d3-outer):
                    //   out_A = [rr | ri] , out_B = [ir | ii] , each half = one real sub-tile.
                    uint8_t* ptr_rr = ptr_outA;
                    uint8_t* ptr_ri = ptr_outA + M3_length_d_tile;
                    uint8_t* ptr_ir = ptr_outB;
                    uint8_t* ptr_ii = ptr_outB + M3_length_d_tile;

                    // ---- SIMD fuse for this tile (both sides, 2 launches each) ----
                    for (int side = 0; side < 2; side++) {
                        uint8_t* pos     = (side == 0) ? ptr_rr : ptr_ri;
                        uint8_t* neg     = (side == 0) ? ptr_ii : ptr_ir;
                        uint16_t* bias_b = ((side == 0) ? ptr_b_re_pp[bbuf] : ptr_b_im_pp[bbuf]);
                        // tile slice of the mini bias (d3-major, contiguous)
                        bias_b             = (uint16_t*)((uint8_t*)bias_b + tile * M3_length_bias_mini_tile);
                        uint8_t* out_t     = (side == 0) ? ptr_out_re_pp[obuf] : ptr_out_im_pp[obuf];
                        uint32_t binop_fp8 = (side == 0) ? M3_SIMD_SUB_FP8_REQUANT : M3_SIMD_ADD_FP8_REQUANT;

                        // pass1': FP8 binop + requant: pos (+/-) neg -> bf16_a
                        if (side == 0) {
                            // Streamers preloaded during OSGEMM B
                            set_simbacore_csr(binop_fp8, seqLen, M3_dPerB_tile, M3_dPerB_tile, 1, 1);
                        } else {
                            set_simd_streamer_csr((uint32_t)pos, M3_R7_widen_ss, M3_R7_widen_tb, M3_R7_widen_ts,
                                                  (uint32_t)neg, M3_R13_fp8_ss, M3_R13_fp8_tb, M3_R13_fp8_ts,
                                                  (uint32_t)ptr_bf16_a, M3_W3_widen_ss, M3_W3_widen_tb,
                                                  M3_W3_widen_ts);
                            write_csr(MODE, binop_fp8);
                        }
                        _set_streamer_start();
                        _set_simbacore_start();
                        write_csr(STREAMER_START_CSR, 0);
                        write_csr(SIMBACORE_START, 0);
                        while (read_csr(SIMBACORE_BUSY));
                        while (read_csr(STREAMER_BUSY_CSR));
                        simbacore_cycles += read_simbacore_perf_counter();

                        // pass2': BF16 ADD bias + requant (+ ReLU in layer 1) -> out tile (FP8)
                        set_simd_streamer_csr((uint32_t)ptr_bf16_a, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts,
                                              (uint32_t)bias_b, M3_R13_bias_ss, M3_R13_bias_tb, M3_R13_bias_ts,
                                              (uint32_t)out_t, M3_W3_fp8_ss, M3_W3_fp8_tb, M3_W3_fp8_ts);
                        write_csr(MODE, add_bias_mode);
                        _set_streamer_start();
                        _set_simbacore_start();
                        write_csr(STREAMER_START_CSR, 0);
                        write_csr(SIMBACORE_START, 0);
                        while (read_csr(SIMBACORE_BUSY));
                        while (read_csr(STREAMER_BUSY_CSR));
                        simbacore_cycles += read_simbacore_perf_counter();
                    }
                    r13_safe_disable();
                }

                snrt_cluster_hw_barrier();
            }

            // Spill the last tile's output (not covered by the in-loop spill).
            if (snrt_is_dm_core()) {
                uint32_t st = nb_tiles - 1;
                int sbuf    = st & 1;
                snrt_dma_start_1d(layer_out_re_l3 + b * M3_length_out_branch + st * M3_length_d_tile,
                                  ptr_out_re_pp[sbuf], M3_length_d_tile);
                snrt_dma_start_1d(layer_out_im_l3 + b * M3_length_out_branch + st * M3_length_d_tile,
                                  ptr_out_im_pp[sbuf], M3_length_d_tile);
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

        err += check_result_sample(l3_out_l1_re, M3_output_1_real, M3_test_samples_output_1_real, nb_test_samples,
                                   "l1_real");
        err += check_result_sample(l3_out_l1_im, M3_output_1_imag, M3_test_samples_output_1_imag, nb_test_samples,
                                   "l1_imag");
        err += check_result_sample(l3_out_l2_re, M3_output_2_real, M3_test_samples_output_2_real, nb_test_samples,
                                   "l2_real");
        err += check_result_sample(l3_out_l2_im, M3_output_2_imag, M3_test_samples_output_2_imag, nb_test_samples,
                                   "l2_imag");
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 4 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}
