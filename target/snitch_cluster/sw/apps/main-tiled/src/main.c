// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// dInner-tiled, double-buffered Mamba main (Phase 1 then Phase 2) with x staged through L3.
// See docs/dataflow/04_mamba_main.md §4.2 for dataflow and per-phase CSR rewrites.
//
// What stays FULL in TCDM: oscore_in, iscore_out_P1 (=dt_in for P2), z, y, iscore_out_P2.
// What is tiled (small TCDM slot, FULL backing in L3): conv_out (=x for P2).
//
// Phase 1 pipeline (P1 has +1 iter for delayed DMA-out):
//   iter i :  (DM)   load weight tile  i
//             (kern) run tile i-1, write to conv_out_tile[(i-1)%2]
//             (DM)   spill conv_out_tile[(i-2)%2] to L3 (tile i-2)
// Phase 2 pipeline:
//   iter i :  (DM)   load weight tile i  AND  DMA-in x_tile[i%2] from L3 (tile i)
//             (kern) run tile i-1, R9 reads x_tile[(i-1)%2]
//
// iscore_out_P1, iscore_out_P2, z, y still stay FULL — see datagen.py TCDM assertion
// and params_in.hjson for the (seqLen, nb_tiles) combinations this layout supports.

#include "helper.c"
#include "snax-simbacore-lib.h"

int test_phase1_and_2() {
    int err = 0;

    // ---- L3 staging buffers. Reserve 16 KiB at the L3 base to skip putc_buffer
    // (see memory note about snrt_l3alloc/putc_buffer overlap).
    //   conv_out_l3 : P1 spills here, P2 fetches into x_tile slots.
    //   z_l3, y_l3  : P2 spills here per-tile (z and y stay tile-sized in TCDM, full in L3).
    static uint8_t* ptr_conv_out_l3 = NULL;
    static uint8_t* ptr_z_l3        = NULL;
    static uint8_t* ptr_y_l3        = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_conv_out_l3 = (uint8_t*)snrt_l3alloc(M1_length_conv_out);
        ptr_z_l3        = (uint8_t*)snrt_l3alloc(M2_length_z);
        ptr_y_l3        = (uint8_t*)snrt_l3alloc(M2_length_y);
    }
    snrt_cluster_hw_barrier();

    // ---- FULL TCDM buffers.
    void* tcdm_base_ptr = snrt_l1_next();

    // Memory layout rationale (split-W3, P1_psum/P2 overlay, L3-staged x/z/y):
    // docs/dataflow/04_mamba_main.md §main-tiled "Memory-saving tricks".
    uint8_t* ptr_oscore_in            = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_iscore_out_P1_psum   = ptr_oscore_in + M1_length_oscore_in;
    uint16_t* ptr_iscore_out_P2       = (uint16_t*)ptr_iscore_out_P1_psum;
    // iscore_out_P1_final follows max(psum, P2) so neither stomps it in its phase.
    uint32_t iscore_shared_bytes      =
        (M1_length_iscore_out > M2_length_iscore_out) ? M1_length_iscore_out : M2_length_iscore_out;
    uint8_t* ptr_iscore_out_P1_final  = ptr_iscore_out_P1_psum + iscore_shared_bytes;

    // P2 consumes the FINAL (transposed) iscore_out_P1 as its dt+BC source.
    uint8_t* ptr_dt_in = ptr_iscore_out_P1_final;
    uint8_t* ptr_BC    = ptr_dt_in + M2_dt_to_BC_offset;

    // R0/R1/R12/R13/W3 need 32 B-aligned base ptrs (sparse interconnect granularity = 4 banks).
    // Use 64 B to also match the AXI DMA burst width.
#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

    uint8_t* pingpong_base_ptr = _ALIGN64(ptr_iscore_out_P1_final + M1_length_iscore_out);

    // ---- Phase 1 ping-pong slots. Last entry (conv_out_tile) is the W1 destination.
    uint8_t* ptr_oscore_weight_P1[2] = {
        pingpong_base_ptr,
        _ALIGN64(pingpong_base_ptr + M1_length_oscore_weight_tile),
    };
    uint8_t* ptr_conv_weight[2] = {
        _ALIGN64(ptr_oscore_weight_P1[1] + M1_length_oscore_weight_tile),
        _ALIGN64(_ALIGN64(ptr_oscore_weight_P1[1] + M1_length_oscore_weight_tile) + M1_length_conv_weight_tile),
    };
    uint8_t* ptr_conv_bias[2] = {
        _ALIGN64(ptr_conv_weight[1] + M1_length_conv_weight_tile),
        _ALIGN64(_ALIGN64(ptr_conv_weight[1] + M1_length_conv_weight_tile) + M1_length_conv_bias_tile),
    };
    uint8_t* ptr_iscore_weight_P1[2] = {
        _ALIGN64(ptr_conv_bias[1] + M1_length_conv_bias_tile),
        _ALIGN64(_ALIGN64(ptr_conv_bias[1] + M1_length_conv_bias_tile) + M1_length_iscore_weight_tile),
    };
    uint8_t* ptr_conv_out_tile[2] = {
        _ALIGN64(ptr_iscore_weight_P1[1] + M1_length_iscore_weight_tile),
        _ALIGN64(_ALIGN64(ptr_iscore_weight_P1[1] + M1_length_iscore_weight_tile) + M1_length_conv_out_tile),
    };

    // ---- Phase 2 ping-pong slots. Overlays Phase 1's ping-pong region.
    uint8_t* ptr_oscore_weight_P2[2] = {
        pingpong_base_ptr,
        _ALIGN64(pingpong_base_ptr + M2_length_oscore_weight_tile),
    };
    uint8_t* ptr_dt_weight_1[2] = {
        _ALIGN64(ptr_oscore_weight_P2[1] + M2_length_oscore_weight_tile),
        _ALIGN64(_ALIGN64(ptr_oscore_weight_P2[1] + M2_length_oscore_weight_tile) + M2_length_dt_weight_1_tile),
    };
    uint8_t* ptr_dt_weight_2[2] = {
        _ALIGN64(ptr_dt_weight_1[1] + M2_length_dt_weight_1_tile),
        _ALIGN64(_ALIGN64(ptr_dt_weight_1[1] + M2_length_dt_weight_1_tile) + M2_length_dt_weight_2_tile),
    };
    uint8_t* ptr_dt_bias[2] = {
        _ALIGN64(ptr_dt_weight_2[1] + M2_length_dt_weight_2_tile),
        _ALIGN64(_ALIGN64(ptr_dt_weight_2[1] + M2_length_dt_weight_2_tile) + M2_length_dt_bias_tile),
    };
    uint8_t* ptr_A[2] = {
        _ALIGN64(ptr_dt_bias[1] + M2_length_dt_bias_tile),
        _ALIGN64(_ALIGN64(ptr_dt_bias[1] + M2_length_dt_bias_tile) + M2_length_A_tile),
    };
    uint8_t* ptr_D[2] = {
        _ALIGN64(ptr_A[1] + M2_length_A_tile),
        _ALIGN64(_ALIGN64(ptr_A[1] + M2_length_A_tile) + M2_length_D_tile),
    };
    uint8_t* ptr_iscore_weight_P2[2] = {
        _ALIGN64(ptr_D[1] + M2_length_D_tile),
        _ALIGN64(_ALIGN64(ptr_D[1] + M2_length_D_tile) + M2_length_iscore_weight_tile),
    };
    uint8_t* ptr_x_tile[2] = {
        _ALIGN64(ptr_iscore_weight_P2[1] + M2_length_iscore_weight_tile),
        _ALIGN64(_ALIGN64(ptr_iscore_weight_P2[1] + M2_length_iscore_weight_tile) + M2_length_x_tile),
    };
    // z_tile and y_tile ping-pong: written by W0/W2 within a kernel, read by R10/R11
    // within the same kernel, then DMA'd out to L3 after the kernel completes.
    uint8_t* ptr_z_tile[2] = {
        _ALIGN64(ptr_x_tile[1] + M2_length_x_tile),
        _ALIGN64(_ALIGN64(ptr_x_tile[1] + M2_length_x_tile) + M2_length_z_tile),
    };
    uint8_t* ptr_y_tile[2] = {
        _ALIGN64(ptr_z_tile[1] + M2_length_z_tile),
        _ALIGN64(_ALIGN64(ptr_z_tile[1] + M2_length_z_tile) + M2_length_y_tile),
    };
#undef _ALIGN64

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Preload Phase 1 non-tiled inputs. Bias goes into the psum buffer (where R13 reads).
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_in, M1_oscore_in, M1_length_oscore_in);
        snrt_dma_start_1d(ptr_iscore_out_P1_psum, M1_iscore_bias, M1_length_iscore_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phase1 = 0;
    uint32_t simbacore_cycles_phase2 = 0;

    // Phase 1.
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: Mamba main tiled (Phase1 then Phase2, nb_tiles=%d, x-tiled via L3)\n\n", nb_tiles);
        start_cycles = snrt_mcycle();
        // Streamer phase1 setter takes a single iscore_out base for both R13 and W3. R13 reads
        // psum_buf always; W3's base is rewritten per-tile in the loop below to redirect the
        // final tile's TRANSPOSE output into ptr_iscore_out_P1_final.
        set_streamer_phase1((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P1[0],     //
                            (uint32_t)ptr_conv_weight[0], (uint32_t)ptr_conv_bias[0],       //
                            (uint32_t)ptr_iscore_weight_P1[0], (uint32_t)ptr_iscore_out_P1_psum,
                            (uint32_t)ptr_conv_out_tile[0]);
        set_simbacore_csr(M1_PHASE1, seqLen, dModel, M1_dInner_tile, dtRank, xProjDim);
    }
    snrt_cluster_hw_barrier();

    // 3-stage pipeline: iter i loads weights[i], runs tile i-1, spills tile i-2 conv_out to L3.
    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int buf = i % 2;

        if (i < nb_tiles && snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_oscore_weight_P1[buf], M1_oscore_weight + i * M1_length_oscore_weight_tile,
                              M1_length_oscore_weight_tile);
            snrt_dma_start_1d(ptr_conv_weight[buf], M1_conv_weight + i * M1_length_conv_weight_tile,
                              M1_length_conv_weight_tile);
            snrt_dma_start_1d(ptr_conv_bias[buf], M1_conv_bias + i * M1_length_conv_bias_tile,
                              M1_length_conv_bias_tile);
            snrt_dma_start_1d(ptr_iscore_weight_P1[buf], M1_iscore_weight + i * M1_length_iscore_weight_tile,
                              M1_length_iscore_weight_tile);
        }

        if (i >= 1 && i <= nb_tiles && snrt_global_core_idx() == 0) {
            uint32_t tile = i - 1;
            int cbuf      = tile % 2;
            write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_oscore_weight_P1[cbuf]);
            write_csr(BASE_PTR_READER_3_LOW, (uint32_t)ptr_conv_weight[cbuf]);
            write_csr(BASE_PTR_READER_4_LOW, (uint32_t)ptr_conv_bias[cbuf]);
            write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_iscore_weight_P1[cbuf]);
            // W1 writes to the ping-pong slot, not into a FULL conv_out. The DMA-out
            // two iterations later moves this slot to L3 at the right tile offset.
            write_csr(BASE_PTR_WRITER_1_LOW, (uint32_t)ptr_conv_out_tile[cbuf]);
            // W3 redirect on the FINAL K-tile: TRANSPOSE output must go to a SEPARATE buffer
            // so it doesn't overwrite psum_partial(n,m) before IS-core processes element (n,m).
            // R13 always reads psum_buf, so non-final K-tiles must still write to psum_buf.
            write_csr(BASE_PTR_WRITER_3_LOW, (uint32_t)((tile == nb_tiles - 1) ? ptr_iscore_out_P1_final
                                                                                : ptr_iscore_out_P1_psum));
            write_csr(MODE, (tile == nb_tiles - 1) ? M1_PHASE1 : M28_PHASE1_NO_REQUANT);
            start_simbacore_and_streamers(M1_R10_en, 0, M1_R11_en, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phase1 += read_simbacore_perf_counter();
            printf("[%u cc] P1 tile %u/%d done\n", snrt_mcycle(), tile + 1, nb_tiles);
        }

        // Spill conv_out_tile produced TWO iterations ago (kernel that ran last iter is
        // already past the barrier, so its conv_out_tile slot is stable here).
        if (i >= 2 && snrt_is_dm_core()) {
            uint32_t spill_tile = i - 2;
            int sbuf            = spill_tile % 2;
            snrt_dma_start_1d(ptr_conv_out_l3 + spill_tile * M1_length_conv_out_tile, ptr_conv_out_tile[sbuf],
                              M1_length_conv_out_tile);
        }

        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    }

    if (snrt_global_core_idx() == 0) printf("[%u cc] P1 done, starting P2 bias preload\n", snrt_mcycle());

    // Preload P2 bias (psum init). oscore_in and dt_in (= iscore_out_P1) are already live.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_iscore_out_P2, M2_iscore_bias, M2_length_iscore_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // Phase 2.
    if (snrt_global_core_idx() == 0) {
        set_streamer_phase2((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P2[0],                     //
                            (uint32_t)ptr_z_tile[0], (uint32_t)ptr_dt_in,                                   //
                            (uint32_t)ptr_dt_weight_1[0], (uint32_t)ptr_dt_weight_2[0],                     //
                            (uint32_t)ptr_dt_bias[0],                                                       //
                            (uint32_t)ptr_x_tile[0], (uint32_t)ptr_A[0], (uint32_t)ptr_BC, (uint32_t)ptr_D[0],
                            (uint32_t)ptr_y_tile[0], (uint32_t)ptr_iscore_weight_P2[0],
                            (uint32_t)ptr_iscore_out_P2);
        set_simbacore_csr(M2_PHASE2, seqLen, dModel, M2_dInner_tile, dtRank, dModel);
    }
    snrt_cluster_hw_barrier();

    // 3-stage pipeline: iter i loads weights+x for tile i, runs tile i-1, spills tile i-2's
    // z and y from TCDM ping-pong slots to L3 (where verification reads them).
    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int buf = i % 2;

        if (i < nb_tiles && snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_oscore_weight_P2[buf], M2_oscore_weight + i * M2_length_oscore_weight_tile,
                              M2_length_oscore_weight_tile);
            snrt_dma_start_1d(ptr_dt_weight_1[buf], M2_dt_weight_1 + i * M2_length_dt_weight_1_tile,
                              M2_length_dt_weight_1_tile);
            snrt_dma_start_1d(ptr_dt_weight_2[buf], M2_dt_weight_2 + i * M2_length_dt_weight_2_tile,
                              M2_length_dt_weight_2_tile);
            snrt_dma_start_1d(ptr_dt_bias[buf], M2_dt_bias + i * M2_length_dt_bias_tile, M2_length_dt_bias_tile);
            snrt_dma_start_1d(ptr_A[buf], M2_suc_A + i * M2_length_A_tile, M2_length_A_tile);
            snrt_dma_start_1d(ptr_D[buf], M2_suc_D + i * M2_length_D_tile, M2_length_D_tile);
            snrt_dma_start_1d(ptr_iscore_weight_P2[buf], M2_iscore_weight + i * M2_length_iscore_weight_tile,
                              M2_length_iscore_weight_tile);
            // Pull this tile's x slice back from L3 (where P1 spilled it).
            // M1_length_conv_out_tile == M2_length_x_tile (same FP8, same L*dInner_tile bytes).
            snrt_dma_start_1d(ptr_x_tile[buf], ptr_conv_out_l3 + i * M2_length_x_tile, M2_length_x_tile);
        }

        if (i >= 1 && i <= nb_tiles && snrt_global_core_idx() == 0) {
            uint32_t tile = i - 1;
            int cbuf      = tile % 2;

            write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_oscore_weight_P2[cbuf]);
            write_csr(BASE_PTR_READER_3_LOW, (uint32_t)ptr_dt_weight_1[cbuf]);
            write_csr(BASE_PTR_READER_4_LOW, (uint32_t)ptr_dt_bias[cbuf]);
            write_csr(BASE_PTR_READER_5_LOW, (uint32_t)ptr_dt_weight_2[cbuf]);
            write_csr(BASE_PTR_READER_6_LOW, (uint32_t)ptr_A[cbuf]);
            write_csr(BASE_PTR_READER_8_LOW, (uint32_t)ptr_D[cbuf]);
            write_csr(BASE_PTR_READER_9_LOW, (uint32_t)ptr_x_tile[cbuf]);
            // z_tile and y_tile become ping-pong slots. W0 writes z_tile, R10 (SUC z input)
            // reads from the SAME slot — both within this kernel. Same for y/W2/R11.
            write_csr(BASE_PTR_READER_10_LOW, (uint32_t)ptr_z_tile[cbuf]);
            write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_y_tile[cbuf]);
            write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_iscore_weight_P2[cbuf]);
            write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_z_tile[cbuf]);
            write_csr(BASE_PTR_WRITER_2_LOW, (uint32_t)ptr_y_tile[cbuf]);

            write_csr(MODE, (tile == nb_tiles - 1) ? M2_PHASE2 : M29_PHASE2_NO_REQUANT);
            start_simbacore_and_streamers(M2_R10_en, M2_R10_start_cnt, M2_R11_en, M2_R11_start_cnt);
            wait_simbacore_and_streamer();
            simbacore_cycles_phase2 += read_simbacore_perf_counter();
            printf("[%u cc] P2 tile %u/%d done\n", snrt_mcycle(), tile + 1, nb_tiles);
        }

        // Spill z and y produced two iterations ago: kernel ran in iter i-1 (so tile i-2),
        // last iter's barrier guarantees the slot is stable.
        if (i >= 2 && snrt_is_dm_core()) {
            uint32_t spill_tile = i - 2;
            int sbuf            = spill_tile % 2;
            snrt_dma_start_1d(ptr_z_l3 + spill_tile * M2_length_z_tile, ptr_z_tile[sbuf], M2_length_z_tile);
            snrt_dma_start_1d(ptr_y_l3 + spill_tile * M2_length_y_tile, ptr_y_tile[sbuf], M2_length_y_tile);
        }

        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    }

    if (snrt_global_core_idx() == 0) printf("[%u cc] P2 done, starting verification\n", snrt_mcycle());

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore Phase1 (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phase1);
        printf("[%d cc] Simbacore Phase2 (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phase2);
        printf("[%d cc] Simbacore total elapsed time: %u cycles\n", end_cycles,
               simbacore_cycles_phase1 + simbacore_cycles_phase2);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        // P1 outputs (inputs to P2/SUC) — checking these first isolates whether bad y
        // is caused by bad x (= conv_out) or bad dt+BC (= iscore_out_P1) rather than a SUC bug.
        err += check_result_sample(ptr_conv_out_l3, M1_conv_out, M1_test_samples_conv_out,  //
                                   nb_test_samples, "P1 conv_out (= P2 x, from L3)");
        err += check_result_sample(ptr_iscore_out_P1_final, M1_iscore_out, M1_test_samples_iscore_out,
                                   nb_test_samples, "P1 iscore_out (= P2 dt+BC)");

        // Full memcmp on P1 conv_out: walks every byte and prints the first 16 mismatches.
        {
            uint32_t mismatches = 0;
            uint32_t conv_len   = seqLen * dInner;  // FP8, 1 byte/elt
            for (uint32_t i = 0; i < conv_len && mismatches < 16; i++) {
                uint8_t got = ptr_conv_out_l3[i];
                uint8_t ref = M1_conv_out[i];
                if (got != ref && !((got == 0 && ref == 128) || (got == 128 && ref == 0))) {
                    printf("DBG conv_out[%u] = %d, ref = %d, diff %d\n", i, got, ref, (int)got - (int)ref);
                    mismatches++;
                }
            }
            printf("DBG conv_out first-16 scan: %u mismatches reported (FP8, total %u bytes)\n",
                   mismatches, conv_len);
        }
        // Same for iscore_out_P1_final (the post-transpose buffer that P2 actually consumes).
        {
            uint32_t mismatches = 0;
            uint32_t isc_len    = seqLen * xProjDim * 2;  // BF16
            for (uint32_t i = 0; i < isc_len && mismatches < 16; i++) {
                if (ptr_iscore_out_P1_final[i] != M1_iscore_out[i]) {
                    printf("DBG iscore_out_P1_final[%u] = %d, ref = %d, diff %d\n", i,
                           ptr_iscore_out_P1_final[i], M1_iscore_out[i],
                           (int)ptr_iscore_out_P1_final[i] - (int)M1_iscore_out[i]);
                    mismatches++;
                }
            }
            printf("DBG iscore_out_P1_final first-16 scan: %u mismatches reported (BF16 bytes, total %u bytes)\n",
                   mismatches, isc_len);
        }

        // z and y now live in L3 (P2 spills them per-tile). check_result_sample reads via
        // AXI directly from L3 — slow per-byte, but only 25 samples, so cost is negligible.
        err += check_result_sample(ptr_z_l3, M2_oscore_expected, M2_test_samples_z,  //
                                   nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y,  //
                                   nb_test_samples, "SUC y");
        err += check_result_sample((uint8_t*)ptr_iscore_out_P2, M2_iscore_expected,  //
                                   M2_test_samples_iscore_out, nb_test_samples, "iscore_out");

        printf("Test Phase1+Phase2 tiled: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d\n",  //
               seqLen, dModel, dInner, nb_tiles);
        // Checks: conv_out, iscore_out_P1, z, y, iscore_out_P2 = 5 * nb_test_samples.
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 5 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() {
    int err = test_phase1_and_2();
    return err;
}
