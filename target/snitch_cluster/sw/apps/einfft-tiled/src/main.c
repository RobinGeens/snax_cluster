// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled EinFFT with PER-TILE fuse.
//   - The N (= D/4 output-channel) axis is tiled. Each tile's 4 OSGEMMs are
//     immediately followed by the SIMD fuse for that tile, so the OS-core
//     scratch (rr/ii/ri/ir), BF16 staging, and output buffers only ever hold
//     ONE tile (L x dPerB_tile) instead of a full per-branch slab. This is
//     what lets large (L, dModel) fit in TCDM.
//   - Weight tiles are prefetched one tile ahead (ping-pong).
//   - Output tiles are spilled to L3 one tile behind (depth-2 ping-pong),
//     overlapping the spill DMA with the next tile's compute.

#include "data.h"
#include "snax-simbacore-lib.h"

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

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

    // Weight ping-pong tiles
    uint8_t* ptr_W_re_pp[2] = {p_after, _ALIGN64(p_after + M3_length_w_branch_tile)};
    p_after                 = _ALIGN64(ptr_W_re_pp[1] + M3_length_w_branch_tile);
    uint8_t* ptr_W_im_pp[2] = {p_after, _ALIGN64(p_after + M3_length_w_branch_tile)};
    p_after                 = _ALIGN64(ptr_W_im_pp[1] + M3_length_w_branch_tile);

    // OS-core scratches + BF16 staging: TILE-sized.
    uint8_t* ptr_rr      = p_after;
    p_after              = _ALIGN64(p_after + M3_length_d_tile);
    uint8_t* ptr_ii      = p_after;
    p_after              = _ALIGN64(p_after + M3_length_d_tile);
    uint8_t* ptr_ri      = p_after;
    p_after              = _ALIGN64(p_after + M3_length_d_tile);
    uint8_t* ptr_ir      = p_after;
    p_after              = _ALIGN64(p_after + M3_length_d_tile);
    uint16_t* ptr_bf16_a = (uint16_t*)p_after;
    p_after              = _ALIGN64(p_after + M3_length_bf16);
    uint16_t* ptr_bf16_b = (uint16_t*)p_after;

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
        uint32_t add_bias_mode   = (layer == 0) ? M3_SIMD_ADD_BF16_RELU : M8_SIMD_ADD_BF16;

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
                        snrt_dma_start_1d(ptr_W_re_pp[buf], w_re_b_l3 + i * M3_length_w_branch_tile,
                                          M3_length_w_branch_tile);
                        snrt_dma_start_1d(ptr_W_im_pp[buf], w_im_b_l3 + i * M3_length_w_branch_tile,
                                          M3_length_w_branch_tile);
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

                    // ---- OSGEMM: rr, ir, ii, ri for this tile ----
                    set_osgemm_streamer_csr((uint32_t)ptr_x_re_b, M3_R0_ss, M3_R0_tb, M3_R0_ts,
                                            (uint32_t)ptr_W_re_pp[cbuf], M3_R1_ss, M3_R1_tb, M3_R1_ts, (uint32_t)ptr_rr,
                                            M3_W0_ss, M3_W0_tb, M3_W0_ts);
                    set_simbacore_csr(M3_OSGEMM, seqLen, M3_dPerB, M3_dPerB_tile, 1, 1);

                    // rr = x_re @ W_re ; rebind for ir = x_im @ W_re
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    write_csr(BASE_PTR_READER_0_LOW, (uint32_t)ptr_x_im_b);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_ir);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // ir ; rebind for ii = x_im @ W_im
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_W_im_pp[cbuf]);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_ii);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // ii ; rebind for ri = x_re @ W_im
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    write_csr(BASE_PTR_READER_0_LOW, (uint32_t)ptr_x_re_b);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_ri);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // ri ; preload SIMD widen streamers for side 0 (fuse follows)
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    set_simd_streamer_no_b((uint32_t)ptr_rr, M3_R7_widen_ss, M3_R7_widen_tb, M3_R7_widen_ts,
                                           (uint32_t)ptr_bf16_a, M3_W3_widen_ss, M3_W3_widen_tb, M3_W3_widen_ts);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // ---- SIMD fuse for this tile (both sides) ----
                    for (int side = 0; side < 2; side++) {
                        uint8_t* pos     = (side == 0) ? ptr_rr : ptr_ri;
                        uint8_t* neg     = (side == 0) ? ptr_ii : ptr_ir;
                        uint16_t* bias_b = ((side == 0) ? ptr_b_re_pp[bbuf] : ptr_b_im_pp[bbuf]);
                        // tile slice of the mini bias (d3-major, contiguous)
                        bias_b         = (uint16_t*)((uint8_t*)bias_b + tile * M3_length_bias_mini_tile);
                        uint8_t* out_t = (side == 0) ? ptr_out_re_pp[obuf] : ptr_out_im_pp[obuf];
                        uint32_t binop = (side == 0) ? M9_SIMD_SUB_BF16 : M8_SIMD_ADD_BF16;

                        // Widen pos -> bf16_a
                        if (side == 0) {
                            // Streamers preloaded during last OSGEMM ri
                            set_simbacore_csr(M25_SIMD_NOOP_FP8_REQUANT, seqLen, M3_dPerB_tile, M3_dPerB_tile, 1, 1);
                        } else {
                            set_simd_streamer_no_b((uint32_t)pos, M3_R7_widen_ss, M3_R7_widen_tb, M3_R7_widen_ts,
                                                   (uint32_t)ptr_bf16_a, M3_W3_widen_ss, M3_W3_widen_tb,
                                                   M3_W3_widen_ts);
                            write_csr(MODE, M25_SIMD_NOOP_FP8_REQUANT);
                        }

                        _set_streamer_start();
                        _set_simbacore_start();
                        write_csr(STREAMER_START_CSR, 0);
                        write_csr(SIMBACORE_START, 0);
                        write_csr(BASE_PTR_READER_7_LOW, (uint32_t)neg);
                        write_csr(BASE_PTR_WRITER_3_LOW, (uint32_t)ptr_bf16_b);
                        while (read_csr(SIMBACORE_BUSY));
                        while (read_csr(STREAMER_BUSY_CSR));
                        simbacore_cycles += read_simbacore_perf_counter();

                        // Widen neg -> bf16_b ; prep binop streamers
                        _set_streamer_start();
                        _set_simbacore_start();
                        write_csr(STREAMER_START_CSR, 0);
                        write_csr(SIMBACORE_START, 0);
                        set_simd_streamer_csr((uint32_t)ptr_bf16_a, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts,
                                              (uint32_t)ptr_bf16_b, M3_R13_bf16_ss, M3_R13_bf16_tb, M3_R13_bf16_ts,
                                              (uint32_t)ptr_bf16_a, M3_W3_bf16_ss, M3_W3_bf16_tb, M3_W3_bf16_ts);
                        while (read_csr(SIMBACORE_BUSY));
                        while (read_csr(STREAMER_BUSY_CSR));
                        simbacore_cycles += read_simbacore_perf_counter();

                        // BF16 binop: bf16_a (+/-) bf16_b -> bf16_a
                        write_csr(MODE, binop);
                        _set_streamer_start();
                        _set_simbacore_start();
                        write_csr(STREAMER_START_CSR, 0);
                        write_csr(SIMBACORE_START, 0);
                        while (read_csr(SIMBACORE_BUSY));
                        while (read_csr(STREAMER_BUSY_CSR));
                        simbacore_cycles += read_simbacore_perf_counter();

                        // Configure R13 mini-bias walk (tile slice)
                        write_csr(BASE_PTR_READER_13_LOW, (uint32_t)bias_b);
                        write_csr(T_BOUND_BASE_READER_13 + 0, M3_R13_bias_tb[0]);
                        write_csr(T_BOUND_BASE_READER_13 + 1, M3_R13_bias_tb[1]);
                        write_csr(T_BOUND_BASE_READER_13 + 2, M3_R13_bias_tb[2]);
                        write_csr(T_BOUND_BASE_READER_13 + 3, M3_R13_bias_tb[3]);
                        write_csr(T_STRIDE_BASE_READER_13 + 0, M3_R13_bias_ts[0]);
                        write_csr(T_STRIDE_BASE_READER_13 + 1, M3_R13_bias_ts[1]);
                        write_csr(T_STRIDE_BASE_READER_13 + 2, M3_R13_bias_ts[2]);
                        write_csr(T_STRIDE_BASE_READER_13 + 3, M3_R13_bias_ts[3]);

                        // BF16 ADD bias (+ ReLU in layer 1) ; prep narrow streamers
                        write_csr(MODE, add_bias_mode);
                        _set_streamer_start();
                        _set_simbacore_start();
                        write_csr(STREAMER_START_CSR, 0);
                        write_csr(SIMBACORE_START, 0);
                        set_simd_streamer_no_b((uint32_t)ptr_bf16_a, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts,
                                               (uint32_t)out_t, M3_W3_fp8_ss, M3_W3_fp8_tb, M3_W3_fp8_ts);
                        while (read_csr(SIMBACORE_BUSY));
                        while (read_csr(STREAMER_BUSY_CSR));
                        simbacore_cycles += read_simbacore_perf_counter();

                        // Narrow BF16 -> FP8 -> out tile
                        write_csr(MODE, M24_SIMD_NOOP_BF16_REQUANT);
                        _set_streamer_start();
                        _set_simbacore_start();
                        write_csr(STREAMER_START_CSR, 0);
                        write_csr(SIMBACORE_START, 0);
                        while (read_csr(SIMBACORE_BUSY));
                        while (read_csr(STREAMER_BUSY_CSR));
                        simbacore_cycles += read_simbacore_perf_counter();
                    }
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
