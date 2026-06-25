// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Bank-conflict-free parallel OSGEMM + ISGEMM. Same arithmetic as is-osgemm-tiled, but
// every buffer uses the bank-partitioned skip-128 layout so OSGEMM lives in TCDM banks
// 0-15 and ISGEMM in banks 16-31. The two cores' streamers can then never collide.
// Design + rationale: docs/dataflow/20_double_gemm_conflict_free.md

#include "data.h"
#include "snax-simbacore-lib.h"

#define NB_STAGES 3
#define SB 128u  // skip-128 block: low 128 B of every 256 B = banks 0-15, high 128 B = banks 16-31

// Physical footprint of a logical buffer of `len` bytes in the skip-128 layout (len % 128 == 0).
#define SKIP_BYTES(len) (2u * (uint32_t)(len))

// DMA a contiguous L3 buffer into a skip-128 TCDM buffer (each 128 B block on a 256 B centre).
static inline void dma_scatter_128(void* dst_skip, const void* src_contig, uint32_t len) {
    snrt_dma_start_2d(dst_skip, src_contig, SB, 2 * SB, SB, len / SB);
}
// Inverse: gather a skip-128 TCDM buffer back to a contiguous L3 buffer.
static inline void dma_gather_128(void* dst_contig, const void* src_skip, uint32_t len) {
    snrt_dma_start_2d(dst_contig, src_skip, SB, SB, 2 * SB, len / SB);
}

int test_double_gemm_conflict_free() {
    int err = 0;

    static uint8_t* l3_d_os  = NULL;
    static uint8_t* l3_cd_is = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);  // reserve putc_buffer region
        l3_d_os  = (uint8_t*)snrt_l3alloc(M3_length_d);
        l3_cd_is = (uint8_t*)snrt_l3alloc(M4_length_cd);  // CD gathered here for checking
    }
    snrt_cluster_hw_barrier();

    // Two bank-partitioned heaps that OVERLAP the same address region. OSGEMM fills the low
    // 128 B of every 256 B block (addr[7]=0 -> banks 0-15); ISGEMM fills the high 128 B of the
    // SAME blocks (base + 128 -> addr[7]=1 -> banks 16-31). Each buffer occupies 2x its logical
    // size, but the opposite half belongs to the other GEMM rather than being wasted, so the
    // total footprint is 2*max(OS, IS) instead of 2*OS + 2*IS. Since every length is a multiple
    // of 128, 2x is a multiple of 256, so consecutive buffers stay aligned to their half.
    uint32_t os_base = ((uint32_t)snrt_l1_next() + 255u) & ~255u;

    uint8_t* ptr_a_os    = (uint8_t*)os_base;
    uint8_t* ptr_b_os[2] = {
        ptr_a_os + SKIP_BYTES(M3_length_a),
        ptr_a_os + SKIP_BYTES(M3_length_a) + SKIP_BYTES(M3_length_b_tile),
    };
    uint8_t* ptr_d_os[2] = {
        ptr_b_os[1] + SKIP_BYTES(M3_length_b_tile),
        ptr_b_os[1] + SKIP_BYTES(M3_length_b_tile) + SKIP_BYTES(M3_length_d_tile),
    };

    uint32_t is_base     = os_base + SB;  // high half of the SAME blocks (+128 -> addr[7]=1, banks 16-31)
    uint8_t* ptr_a_is[2] = {
        (uint8_t*)is_base,
        (uint8_t*)is_base + SKIP_BYTES(M4_length_a_tile),
    };
    uint8_t* ptr_b_is[2] = {
        ptr_a_is[1] + SKIP_BYTES(M4_length_a_tile),
        ptr_a_is[1] + SKIP_BYTES(M4_length_a_tile) + SKIP_BYTES(M4_length_b_tile),
    };
    uint16_t* ptr_cd_is = (uint16_t*)(ptr_b_is[1] + SKIP_BYTES(M4_length_b_tile));

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Preload (skip-128 scatter): OSGEMM A (shared across tiles) and ISGEMM bias C -> CD accumulator.
    if (snrt_is_dm_core()) {
        dma_scatter_128(ptr_a_os, M3_A, M3_length_a);
        dma_scatter_128((uint8_t*)ptr_cd_is, M4_C, M4_length_cd);
        snrt_dma_wait_all();
    }

    // One-time streamer setup: OSGEMM uses R0/R1/W0, ISGEMM uses R11/R12/R13/W3.
    // The temporal bounds/strides in data.h already encode the skip-128 walk.
    if (snrt_global_core_idx() == 0) {
        set_streamer_csr((uint32_t)ptr_a_os, M3_R0_ss, M3_R0_tb, M3_R0_ts, M3_R0_en,         // R0: oscore A
                         (uint32_t)ptr_b_os[0], M3_R1_ss, M3_R1_tb, M3_R1_ts, M3_R1_en,      // R1: oscore B
                         (uint32_t)0, 0, 0, 0, 0,                                            // R2: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R3: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R4: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R5: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R6: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R7: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R8: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R9: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R10: disabled
                         (uint32_t)ptr_a_is[0], M4_R11_ss, M4_R11_tb, M4_R11_ts, M4_R11_en,  // R11: iscore A
                         (uint32_t)ptr_b_is[0], M4_R12_ss, M4_R12_tb, M4_R12_ts, M4_R12_en,  // R12: iscore B
                         (uint32_t)ptr_cd_is, M4_R13_ss, M4_R13_tb, M4_R13_ts, M4_R13_en,    // R13: iscore CD
                         (uint32_t)ptr_d_os[0], M3_W0_ss, M3_W0_tb, M3_W0_ts, M3_W0_en,      // W0: oscore D
                         (uint32_t)0, 0, 0, 0, 0,                                            // W1: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // W2: disabled
                         (uint32_t)ptr_cd_is, M4_W3_ss, M4_W3_tb, M4_W3_ts, M4_W3_en         // W3: iscore CD
        );
        set_simbacore_csr(IS_OSGEMM_NO_REQUANT, seqLen, dModel, dInner_tile, 1, dModel);
    }

    snrt_cluster_hw_barrier();

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;
    static uint32_t _dma_done = 0, _compute_done = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: double-gemm-conflict-free (nb_tiles=%d)\n\n", nb_tiles);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        start_cycles = snrt_mcycle();
    }

    // Pipelined loop: nb_tiles + NB_STAGES - 1 iterations.
    //   i in [0, nb_tiles)       -> transfer_in tile i        (DM core)
    //   i in [1, nb_tiles + 1)   -> compute     tile i-1      (core 0)
    //   i in [2, nb_tiles + 2)   -> transfer_out tile i-2     (DM core)
    for (uint32_t i = 0; i < nb_tiles + NB_STAGES - 1; i++) {
        int buf = i % 2;

        // Stage 1: transfer_in B_os tile, A_is tile, B_is tile (skip-128 scatter)
        if (i < nb_tiles) {
            if (snrt_is_dm_core()) {
                dma_scatter_128(ptr_b_os[buf], M3_B + i * M3_length_b_tile, M3_length_b_tile);
                dma_scatter_128(ptr_a_is[buf], M4_A + i * M4_length_a_tile, M4_length_a_tile);
                dma_scatter_128(ptr_b_is[buf], M4_B + i * M4_length_b_tile, M4_length_b_tile);
            }
        }

        // Stage 2: compute tile i-1 (both cores in parallel).
        if (i >= 1 && i < nb_tiles + 1) {
            uint32_t tile = i - 1;
            if (snrt_global_core_idx() == 0) {
                write_csr(MODE, (tile == nb_tiles - 1) ? IS_OSGEMM : IS_OSGEMM_NO_REQUANT);
                start_simbacore_and_streamers(0, 0, M4_R11_en, 0);
                write_csr(STREAMER_START_CSR, 0);
                write_csr(SIMBACORE_START, 0);
                write_csr(DELAYED_START_READER_10, 0);
                write_csr(DELAYED_START_READER_11, 0);

                // CSR pre-load
                if (tile < nb_tiles - 1) {
                    int nbuf = (tile + 1) % 2;
                    write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_b_os[nbuf]);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_d_os[nbuf]);
                    write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_a_is[nbuf]);
                    write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_b_is[nbuf]);
                }
                while (read_csr(SIMBACORE_BUSY));
                while (read_csr(STREAMER_BUSY_CSR));
                simbacore_cycles_total += read_simbacore_perf_counter();
                if (i == 1) _compute_done = snrt_mcycle();
            }
        }

        // Stage 3: spill D_os tile i-2 to L3 (skip-128 gather -> contiguous L3)
        if (i >= 2) {
            uint32_t tile = i - 2;
            int sbuf      = tile % 2;
            if (snrt_is_dm_core()) {
                dma_gather_128(l3_d_os + tile * M3_length_d_tile, ptr_d_os[sbuf], M3_length_d_tile);
            }
        }

        if (snrt_is_dm_core()) {
            snrt_dma_wait_all();
            // First iteration: check if time(DMA) < time(compute) for latency hiding
            if (i == 1) _dma_done = snrt_mcycle();
        }
        snrt_cluster_hw_barrier();
    }

    // Stop the clock here (matches the baseline, which checks its final result in place): the
    // CD gather below is verification-only staging, not part of the kernel.
    uint32_t end_cycles = 0;
    if (snrt_global_core_idx() == 0) end_cycles = snrt_mcycle();

    // CD_is accumulator lives in skip-128 TCDM; gather it to contiguous L3 for checking.
    if (snrt_is_dm_core()) {
        dma_gather_128(l3_cd_is, (uint8_t*)ptr_cd_is, M4_length_cd);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_total);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        printf("DMA latency hiding: tile=%s\n", _dma_done < _compute_done ? "ok" : "STALL");
        err += check_result_sample(l3_d_os, M3_D, M3_test_samples_D, nb_test_samples, "osgemm_out");
        err += check_result_sample(l3_cd_is, M4_D, M4_test_samples_D, nb_test_samples, "isgemm_out");

        printf("Test double-gemm-conflict-free: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d\n", seqLen, dModel, dInner,
               nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_double_gemm_conflict_free(); }
