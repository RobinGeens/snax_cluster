// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled EinFFT implementation.
// Same ConvFormat-throughout strategy as the un-tiled `einfft` (see
// docs/dataflow/06_einfft_mlp.md). The D/4 output-channel axis is
// tiled at the OS-core level only: weights are DMA'd one tile at a time
// (ping-pong against compute) and each per-tile OSGEMM output drops into the
// right slot of the FULL per-branch ConvFormat scratch. The SIMD fuse then
// runs ONCE per side per branch over the assembled scratch.

#include "data.h"
#include "snax-simbacore-lib.h"

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

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

    // Per-(layer, branch) scratch placed right after the FULL inputs. Weight
    // ping-pong slots are per-tile-sized; the OS-core scratches and BF16
    // staging are FULL per-branch.
    uint8_t* pp_base = _ALIGN64((uint8_t*)(tcdm_base_ptr + M3_addr_output_2_imag + M3_length_output_2_imag));

    uint8_t* ptr_W_re_pp[2] = {pp_base, _ALIGN64(pp_base + M3_length_w_branch_tile)};
    uint8_t* p_after_Wre    = _ALIGN64(ptr_W_re_pp[1] + M3_length_w_branch_tile);
    uint8_t* ptr_W_im_pp[2] = {p_after_Wre, _ALIGN64(p_after_Wre + M3_length_w_branch_tile)};
    uint8_t* p_after        = _ALIGN64(ptr_W_im_pp[1] + M3_length_w_branch_tile);

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
            "\nStarting program: einfft-tiled "
            "(L=%u, dModel=%u, dPerB=%u, dPerB_tile=%u, nb_tiles=%u, nBranches=%u)\n\n",
            seqLen, dModel, M3_dPerB, M3_dPerB_tile, nb_tiles, M3_nBranches);
        start_cycles = snrt_mcycle();
    }

    // ==================================================================
    // Two-layer loop. Each layer:
    //   For each branch:
    //     For each N-tile (ping-pong weight DMA against compute):
    //       4 OSGEMMs writing into the per-tile slot of rr/ii/ri/ir.
    //     SIMD widen → SUB/ADD → bias-ADD (+ReLU for L1) → narrow,
    //       once per side, on the FULL per-branch ConvFormat scratch.
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
            uint8_t* x_re_b     = layer_x_re + b * M3_length_x_branch;
            uint8_t* x_im_b     = layer_x_im + b * M3_length_x_branch;
            uint8_t* out_re_b   = layer_out_re + b * M3_length_out_branch;
            uint8_t* out_im_b   = layer_out_im + b * M3_length_out_branch;
            uint8_t* w_re_b_l3  = layer_w_re_l3 + b * M3_length_w_branch;
            uint8_t* w_im_b_l3  = layer_w_im_l3 + b * M3_length_w_branch;
            uint16_t* b_re_bc_b = layer_b_re_bc + b * (M3_length_bias_bcast_branch / 2);
            uint16_t* b_im_bc_b = layer_b_im_bc + b * (M3_length_bias_bcast_branch / 2);

            // ============================================================
            // Per-tile weight DMA + 4 OSGEMMs (2-stage ping-pong pipeline).
            // Iter i loads tile i, compute fires on tile i-1.
            // ============================================================
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

                    // 4 OSGEMMs per tile: rr / ir / ii / ri.
                    set_osgemm_streamer_csr((uint32_t)x_re_b, M3_R0_ss, M3_R0_tb, M3_R0_ts, (uint32_t)ptr_W_re_pp[cbuf],
                                            M3_R1_ss, M3_R1_tb, M3_R1_ts, (uint32_t)rr_tile, M3_W0_ss, M3_W0_tb,
                                            M3_W0_ts);
                    set_simbacore_csr(M3_OSGEMM, seqLen, M3_dPerB, M3_dPerB_tile, 1, 1);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();

                    write_csr(BASE_PTR_READER_0_LOW, (uint32_t)x_im_b);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ir_tile);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();

                    write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_W_im_pp[cbuf]);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ii_tile);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();

                    write_csr(BASE_PTR_READER_0_LOW, (uint32_t)x_re_b);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ri_tile);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();
                }

                if (snrt_is_dm_core()) snrt_dma_wait_all();
                snrt_cluster_hw_barrier();
            }

            // ============================================================
            // Per-branch SIMD fuse (FULL per-branch bounds).
            // ============================================================
            if (snrt_global_core_idx() == 0) {
                for (int side = 0; side < 2; side++) {
                    uint8_t* pos      = (side == 0) ? ptr_rr : ptr_ri;
                    uint8_t* neg      = (side == 0) ? ptr_ii : ptr_ir;
                    uint16_t* bias_b  = (side == 0) ? b_re_bc_b : b_im_bc_b;
                    uint8_t* out_full = (side == 0) ? out_re_b : out_im_b;
                    uint32_t binop    = (side == 0) ? M9_SIMD_SUB_BF16 : M8_SIMD_ADD_BF16;

                    // Widen FP8 → BF16 (pos): rr / ri → bf16_a.
                    set_simd_streamer_no_b((uint32_t)pos, M3_R7_widen_ss, M3_R7_widen_tb, M3_R7_widen_ts,
                                           (uint32_t)ptr_bf16_a, M3_W3_widen_ss, M3_W3_widen_tb, M3_W3_widen_ts);
                    set_simbacore_csr(M25_SIMD_NOOP_FP8_REQUANT, seqLen, M3_dPerB, M3_dPerB, 1, 1);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();

                    // Widen FP8 → BF16 (neg): ii / ir → bf16_b.
                    write_csr(BASE_PTR_READER_7_LOW, (uint32_t)neg);
                    write_csr(BASE_PTR_WRITER_3_LOW, (uint32_t)ptr_bf16_b);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();

                    // BF16 binop: bf16_a (±) bf16_b → bf16_a.
                    set_simd_streamer_csr((uint32_t)ptr_bf16_a, M3_R7_bf16_ss, M3_R7_bf16_tb, M3_R7_bf16_ts,
                                          (uint32_t)ptr_bf16_b, M3_R13_bf16_ss, M3_R13_bf16_tb, M3_R13_bf16_ts,
                                          (uint32_t)ptr_bf16_a, M3_W3_bf16_ss, M3_W3_bf16_tb, M3_W3_bf16_ts);
                    set_simbacore_csr(binop, seqLen, M3_dPerB, M3_dPerB, 1, 1);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();

                    // BF16 ADD bias broadcast: bf16_a + bias_bcast_branch → bf16_a.
                    // Layer 1 fuses doRelu into the mode.
                    write_csr(BASE_PTR_READER_13_LOW, (uint32_t)bias_b);
                    write_csr(MODE, add_bias_mode);
                    start_simbacore_and_streamers(0, 0, 0, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();

                    // Narrow BF16 → FP8: bf16_a → ConvFormat output (per branch).
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
