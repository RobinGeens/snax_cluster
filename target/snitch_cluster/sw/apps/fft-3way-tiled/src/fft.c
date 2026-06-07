// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled 3-way partitioned EinFFT (L = L1 * L2 * L3). dModel is the independent
// batch axis of every partition, so the full un-tiled kernel is run once per
// dModel-slice: every buffer shrinks to 1/nb_d and fits TCDM. Each slice ping-
// pongs through two TCDM slots; the full output is assembled in L3.
// See docs/dataflow/05_fft.md §5.4 for the dataflow and slot/overlay rationale.

#include "../data/data.h"
#include "snax-simbacore-lib.h"

static inline uint32_t align64(uint32_t x) { return (x + 63u) & ~63u; }

int test() {
    int err = 0;

    // L3: reserve 16 KiB past the runtime's putc_buffer (see fft-tiled note on
    // snrt_l3alloc/_edram overlap), then a FULL partition3_out output buffer that
    // the per-slice DMA-out assembles into.
    static uint8_t* ptr_output_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_output_l3 = (uint8_t*)snrt_l3alloc(M6_length_partition3_out);
    }
    snrt_cluster_hw_barrier();

    // TCDM: always-live weights/twiddles, then two ping-pong slots. Every step
    // reads one slot and writes the other; slots are sized to the largest
    // per-slice buffer (partition_out, BF16).
    void* tcdm_base_ptr    = snrt_l1_next();
    uint8_t* ptr_weight1   = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_weight2   = ptr_weight1 + align64(M6_length_weight1);
    uint8_t* ptr_weight3   = ptr_weight2 + align64(M6_length_weight2);
    uint8_t* ptr_twiddles1 = ptr_weight3 + align64(M6_length_weight3);
    uint8_t* ptr_twiddles2 = ptr_twiddles1 + align64(M6_length_twiddles1);
    uint8_t* ptr_slotA     = ptr_twiddles2 + align64(M6_length_twiddles2);
    uint8_t* ptr_slotB     = ptr_slotA + M6_slot_size;

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Weights and twiddles depend only on (l1, l2, l3): loaded once, broadcast
    // across all dModel slices.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_weight3, M6_dft_weight3, M6_length_weight3);
        snrt_dma_start_1d(ptr_twiddles1, M6_twiddles1, M6_length_twiddles1);
        snrt_dma_start_1d(ptr_twiddles2, M6_twiddles2, M6_length_twiddles2);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles     = 0;
    uint32_t simbacore_cycles = 0;
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: tiled 3-way FFT (outer dModel-tile, nb_d=%d, seqLen=%d, dModel=%d, "
               "L1=%d, L2=%d, L3=%d)\n\n",
               M6_nb_d, seqLen, dModel, L1, L2, L3);
        start_cycles = snrt_mcycle();
    }

    for (uint32_t s = 0; s < M6_nb_d; s++) {
        // Load this slice's input → slotA; zero slotB (partition1 psum target).
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_slotA, M6_dft_in + s * M6_length_in_slice, M6_length_in_slice);
            snrt_dma_start_1d(ptr_slotB, (void*)snrt_zero_memory_ptr(), M6_slot_size);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            // Step 1: partition 1 (slotA → slotB).
            set_isgemm_streamer_csr((uint32_t)ptr_weight1, M6_R11_1_ss, M6_R11_1_tb, M6_R11_1_ts,
                                    (uint32_t)ptr_slotA, M6_R12_1_ss, M6_R12_1_tb, M6_R12_1_ts,
                                    (uint32_t)ptr_slotB, M6_W3_1_ss, M6_W3_1_tb, M6_W3_1_ts);
            set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L1, 1, L1_padded, 1, M6_dModel_slice * L2 * L3);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Step 2: hadamard 1 CMUL (slotB → slotA).
            set_simd_streamer_csr((uint32_t)ptr_slotB, M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts,
                                  (uint32_t)ptr_twiddles1, M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts,
                                  (uint32_t)ptr_slotA, M6_W3_2_ss, M6_W3_2_tb, M6_W3_2_ts);
            set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Step 2B: reorder 1 NOOP (slotA → slotB).
            set_simd_streamer_no_b((uint32_t)ptr_slotA, M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts,
                                   (uint32_t)ptr_slotB, M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts);
            set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();

        // Zero slotA for partition2 psum (slotA now holds dead hadamard1_out).
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_slotA, (void*)snrt_zero_memory_ptr(), M6_slot_size);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            // Step 3: partition 2 (slotB → slotA).
            set_isgemm_streamer_csr((uint32_t)ptr_weight2, M6_R11_3_ss, M6_R11_3_tb, M6_R11_3_ts,
                                    (uint32_t)ptr_slotB, M6_R12_3_ss, M6_R12_3_tb, M6_R12_3_ts,
                                    (uint32_t)ptr_slotA, M6_W3_3_ss, M6_W3_3_tb, M6_W3_3_ts);
            set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L2, 1, 2 * L2_padded, 1, M6_dModel_slice * L1 * L3);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Step 4: hadamard 2 CMUL (slotA → slotB).
            set_simd_streamer_csr((uint32_t)ptr_slotA, M6_R7_4_ss, M6_R7_4_tb, M6_R7_4_ts,
                                  (uint32_t)ptr_twiddles2, M6_R13_4_ss, M6_R13_4_tb, M6_R13_4_ts,
                                  (uint32_t)ptr_slotB, M6_W3_4_ss, M6_W3_4_tb, M6_W3_4_ts);
            set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Step 4B: reorder 2 NOOP (slotB → slotA).
            set_simd_streamer_no_b((uint32_t)ptr_slotB, M6_R7_4B_ss, M6_R7_4B_tb, M6_R7_4B_ts,
                                   (uint32_t)ptr_slotA, M6_W3_4B_ss, M6_W3_4B_tb, M6_W3_4B_ts);
            set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();

        // Zero slotB for partition3 psum (slotB now holds dead hadamard2_out).
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_slotB, (void*)snrt_zero_memory_ptr(), M6_slot_size);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            // Step 5: partition 3 (slotA → slotB).
            set_isgemm_streamer_csr((uint32_t)ptr_weight3, M6_R11_5_ss, M6_R11_5_tb, M6_R11_5_ts,
                                    (uint32_t)ptr_slotA, M6_R12_5_ss, M6_R12_5_tb, M6_R12_5_ts,
                                    (uint32_t)ptr_slotB, M6_W3_5_ss, M6_W3_5_tb, M6_W3_5_ts);
            set_simbacore_csr(M6_ISGEMM_SQ, 2 * L3, 1, 2 * L3_padded, 1, M6_dModel_slice * L1 * L2);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Ensure slotB is drained to TCDM before the DMA-out reads it.
            asm volatile("fence" ::: "memory");
        }
        snrt_cluster_hw_barrier();

        // Store this slice's partition3_out into its d-slice of the full L3 output.
        // partition3_out's final type is FP8, which the IS-core zero-pads to the BF16
        // psum footprint, so only the first half of slotB holds real data. That real
        // data is M6_out_nblk = M_3 row-blocks with d the outer column factor; slice s
        // owns out_block_slice bytes (dM channels) within each full out_block_full
        // row-block. See docs/dataflow/05_fft.md §5.4.
        if (snrt_is_dm_core()) {
            snrt_dma_start_2d(ptr_output_l3 + s * M6_out_block_slice, ptr_slotB, M6_out_block_slice,
                              /*dst_stride=*/M6_out_block_full, /*src_stride=*/M6_out_block_slice,
                              /*repeat=*/M6_out_nblk);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
    }

    // --- Verification: full output assembled in L3, scalar-read at sample positions. ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_output_l3, M6_partition3_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition3_out (L3)");

        printf("Test FFT 3-way tiled (outer dModel-tile): (%d x %d), L1=%d L2=%d L3=%d, nb_d=%d\n",
               seqLen, dModel, L1, L2, L3, M6_nb_d);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
