// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled EinFFT implementation with L3-staged activations.
// Same ConvFormat-throughout strategy as `einfft`, but only the CURRENT
// (layer, branch) input / bias_bcast / output lives in TCDM — DMA'd in/out
// from L3 per branch. Weights are still tile-ping-pong'd against compute
// along the N (= dPerB) output-channel axis.

#include "data.h"
#include "snax-simbacore-lib.h"

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

int main(void) {
    int err = 0;

    // ---- L3 staging for the 4 layer-outputs. Reserve 16 KiB up-front to skip the putc_buffer overlap at &_edram (see
    // snrt_l3alloc note).
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

    // ---- Per-branch resident TCDM slots ---------------------------------
    uint8_t* ptr_x_re_b     = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_x_im_b     = _ALIGN64(ptr_x_re_b + M3_length_x_branch);
    uint16_t* ptr_b_re_bc_b = (uint16_t*)_ALIGN64(ptr_x_im_b + M3_length_x_branch);
    uint16_t* ptr_b_im_bc_b = (uint16_t*)_ALIGN64((uint8_t*)ptr_b_re_bc_b + M3_length_bias_bcast_branch);
    uint8_t* ptr_out_re_b   = _ALIGN64((uint8_t*)ptr_b_im_bc_b + M3_length_bias_bcast_branch);
    uint8_t* ptr_out_im_b   = _ALIGN64(ptr_out_re_b + M3_length_out_branch);

    uint8_t* p_after = _ALIGN64(ptr_out_im_b + M3_length_out_branch);

    // Weight ping-pong tiles
    uint8_t* ptr_W_re_pp[2] = {p_after, _ALIGN64(p_after + M3_length_w_branch_tile)};
    p_after                 = _ALIGN64(ptr_W_re_pp[1] + M3_length_w_branch_tile);
    uint8_t* ptr_W_im_pp[2] = {p_after, _ALIGN64(p_after + M3_length_w_branch_tile)};
    p_after                 = _ALIGN64(ptr_W_im_pp[1] + M3_length_w_branch_tile);

    // OS-core scratches (FULL per-branch ConvFormat) + BF16 staging.
    uint8_t* ptr_rr      = p_after;
    p_after              = _ALIGN64(p_after + M3_length_out_branch);
    uint8_t* ptr_ii      = p_after;
    p_after              = _ALIGN64(p_after + M3_length_out_branch);
    uint8_t* ptr_ri      = p_after;
    p_after              = _ALIGN64(p_after + M3_length_out_branch);
    uint8_t* ptr_ir      = p_after;
    p_after              = _ALIGN64(p_after + M3_length_out_branch);
    uint16_t* ptr_bf16_a = (uint16_t*)p_after;
    p_after              = _ALIGN64(p_after + 2 * M3_length_out_branch);
    uint16_t* ptr_bf16_b = (uint16_t*)p_after;

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    uint32_t simbacore_cycles = 0, start_cycles = 0;
    static uint32_t _dma_done = 0, _compute_done = 0;

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: einfft-tiled (L3-staged) "
            "(L=%u, dModel=%u, dPerB=%u, dPerB_tile=%u, nb_tiles=%u, nBranches=%u)\n\n",
            seqLen, dModel, M3_dPerB, M3_dPerB_tile, nb_tiles, M3_nBranches);
        start_cycles = snrt_mcycle();
    }

    // ==================================================================
    // Two-layer loop. Per (layer, branch):
    //   DMA-in x_re_b, x_im_b, b_re_bc_b, b_im_bc_b for THIS (layer, branch).
    //   Tile loop: weight ping-pong DMA + 4 OSGEMMs/tile (rr/ir/ii/ri).
    //   Per-side SIMD fuse over the FULL per-branch scratch.
    //   DMA-out out_re_b, out_im_b to L3 output buffer at branch offset.
    // ==================================================================
    for (int layer = 0; layer < 2; layer++) {
        uint8_t* layer_x_re_l3   = (layer == 0) ? M3_x_real : M3_x_2_real;
        uint8_t* layer_x_im_l3   = (layer == 0) ? M3_x_imag : M3_x_2_imag;
        uint8_t* layer_w_re_l3   = (layer == 0) ? M3_weight_1_real : M3_weight_2_real;
        uint8_t* layer_w_im_l3   = (layer == 0) ? M3_weight_1_imag : M3_weight_2_imag;
        uint8_t* layer_b_re_l3   = (layer == 0) ? (uint8_t*)M3_bias_1_real_bcast : (uint8_t*)M3_bias_2_real_bcast;
        uint8_t* layer_b_im_l3   = (layer == 0) ? (uint8_t*)M3_bias_1_imag_bcast : (uint8_t*)M3_bias_2_imag_bcast;
        uint8_t* layer_out_re_l3 = (layer == 0) ? l3_out_l1_re : l3_out_l2_re;
        uint8_t* layer_out_im_l3 = (layer == 0) ? l3_out_l1_im : l3_out_l2_im;
        uint32_t add_bias_mode   = (layer == 0) ? M3_SIMD_ADD_BF16_RELU : M8_SIMD_ADD_BF16;

        for (uint32_t b = 0; b < M3_nBranches; b++) {
            // ---- DMA-in this branch's inputs from L3 -------------------
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_x_re_b, layer_x_re_l3 + b * M3_length_x_branch, M3_length_x_branch);
                snrt_dma_start_1d(ptr_x_im_b, layer_x_im_l3 + b * M3_length_x_branch, M3_length_x_branch);
                snrt_dma_start_1d((uint8_t*)ptr_b_re_bc_b, layer_b_re_l3 + b * M3_length_bias_bcast_branch,
                                  M3_length_bias_bcast_branch);
                snrt_dma_start_1d((uint8_t*)ptr_b_im_bc_b, layer_b_im_l3 + b * M3_length_bias_bcast_branch,
                                  M3_length_bias_bcast_branch);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();

            uint8_t* w_re_b_l3 = layer_w_re_l3 + b * M3_length_w_branch;
            uint8_t* w_im_b_l3 = layer_w_im_l3 + b * M3_length_w_branch;

            // ---- Per-tile weight DMA + 4 OSGEMMs (2-stage ping-pong) ---
            // Iter i loads tile i, compute fires on tile i-1.
            for (uint32_t i = 0; i < nb_tiles + 1; i++) {
                int buf = i & 1;

                if (i < nb_tiles && snrt_is_dm_core()) {
                    snrt_dma_start_1d(ptr_W_re_pp[buf], w_re_b_l3 + i * M3_length_w_branch_tile,
                                      M3_length_w_branch_tile);
                    snrt_dma_start_1d(ptr_W_im_pp[buf], w_im_b_l3 + i * M3_length_w_branch_tile,
                                      M3_length_w_branch_tile);
                }

                if (i >= 1 && snrt_global_core_idx() == 0) {
                    uint32_t tile    = i - 1;
                    int cbuf         = tile & 1;
                    uint8_t* rr_tile = ptr_rr + tile * M3_length_d_tile;
                    uint8_t* ii_tile = ptr_ii + tile * M3_length_d_tile;
                    uint8_t* ri_tile = ptr_ri + tile * M3_length_d_tile;
                    uint8_t* ir_tile = ptr_ir + tile * M3_length_d_tile;

                    // Tile 0: full CSR setup. Tile > 0: base ptrs preloaded from prev tile's ri.
                    if (tile == 0) {
                        set_osgemm_streamer_csr((uint32_t)ptr_x_re_b, M3_R0_ss, M3_R0_tb, M3_R0_ts,
                                                (uint32_t)ptr_W_re_pp[cbuf], M3_R1_ss, M3_R1_tb, M3_R1_ts,
                                                (uint32_t)rr_tile, M3_W0_ss, M3_W0_tb, M3_W0_ts);
                        set_simbacore_csr(M3_OSGEMM, seqLen, M3_dPerB, M3_dPerB_tile, 1, 1);
                    }

                    // rr
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    write_csr(BASE_PTR_READER_0_LOW, (uint32_t)ptr_x_im_b);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ir_tile);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // ir (base ptrs preloaded from rr)
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_W_im_pp[cbuf]);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ii_tile);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // ii (base ptrs preloaded from ir)
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    write_csr(BASE_PTR_READER_0_LOW, (uint32_t)ptr_x_re_b);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ri_tile);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // ri (base ptrs preloaded from ii)
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    if (tile < nb_tiles - 1) {
                        int nbuf = (tile + 1) & 1;
                        write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_W_re_pp[nbuf]);
                        write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)(ptr_rr + (tile + 1) * M3_length_d_tile));
                    }
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    if (i == 1) _compute_done = snrt_mcycle();
                    // printf("[%u cc] L%d B%u tile %u/%d done\n", snrt_mcycle(), layer, b, tile + 1, nb_tiles);
                }

                if (snrt_is_dm_core()) {
                    snrt_dma_wait_all();
                    if (i == 1 && i < nb_tiles) _dma_done = snrt_mcycle();
                }
                snrt_cluster_hw_barrier();
            }

            // ---- Per-branch SIMD fuse (FULL per-branch bounds) ---------
            if (snrt_global_core_idx() == 0) {
                for (int side = 0; side < 2; side++) {
                    uint8_t* pos      = (side == 0) ? ptr_rr : ptr_ri;
                    uint8_t* neg      = (side == 0) ? ptr_ii : ptr_ir;
                    uint16_t* bias_b  = (side == 0) ? ptr_b_re_bc_b : ptr_b_im_bc_b;
                    uint8_t* out_full = (side == 0) ? ptr_out_re_b : ptr_out_im_b;
                    uint32_t binop    = (side == 0) ? M9_SIMD_SUB_BF16 : M8_SIMD_ADD_BF16;

                    // Widen FP8 → BF16 (pos): rr / ri → bf16_a.
                    set_simd_streamer_no_b((uint32_t)pos, M3_R7_widen_ss, M3_R7_widen_tb, M3_R7_widen_ts,
                                           (uint32_t)ptr_bf16_a, M3_W3_widen_ss, M3_W3_widen_tb, M3_W3_widen_ts);
                    if (side == 0)
                        set_simbacore_csr(M25_SIMD_NOOP_FP8_REQUANT, seqLen, M3_dPerB, M3_dPerB, 1, 1);
                    else
                        write_csr(MODE, M25_SIMD_NOOP_FP8_REQUANT);
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    write_csr(BASE_PTR_READER_7_LOW, (uint32_t)neg);
                    write_csr(BASE_PTR_WRITER_3_LOW, (uint32_t)ptr_bf16_b);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // Widen FP8 → BF16 (neg): ii / ir → bf16_b (base ptrs preloaded).
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

                    // BF16 binop: bf16_a (±) bf16_b → bf16_a (streamers preloaded).
                    write_csr(MODE, binop);
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    write_csr(BASE_PTR_READER_13_LOW, (uint32_t)bias_b);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // BF16 ADD bias broadcast → bf16_a (R13 preloaded).
                    write_csr(MODE, add_bias_mode);
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    set_simd_streamer_no_b((uint32_t)ptr_bf16_a, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts,
                                           (uint32_t)out_full, M3_W3_fp8_ss, M3_W3_fp8_tb, M3_W3_fp8_ts);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();

                    // Narrow BF16 → FP8: bf16_a → per-branch out (streamers preloaded).
                    write_csr(MODE, M24_SIMD_NOOP_BF16_REQUANT);
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles += read_simbacore_perf_counter();
                }
                // printf("[%u cc] L%d B%u SIMD done\n", snrt_mcycle(), layer, b);
            }
            snrt_cluster_hw_barrier();

            // ---- DMA-out this branch's output to L3 -------------------
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(layer_out_re_l3 + b * M3_length_out_branch, ptr_out_re_b, M3_length_out_branch);
                snrt_dma_start_1d(layer_out_im_l3 + b * M3_length_out_branch, ptr_out_im_b, M3_length_out_branch);
                snrt_dma_wait_all();
            }

            snrt_cluster_hw_barrier();
        }

        // if (snrt_global_core_idx() == 0) printf("[%u cc] Layer %d done\n", snrt_mcycle(), layer);
    }

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%u cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        printf("DMA latency hiding: wgt_tile=%s\n", _dma_done < _compute_done ? "hidden" : "STALL");

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
