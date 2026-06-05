// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled 3-way partitioned EinFFT (L = L1 * L2 * L3). dModel is the independent
// batch axis of every partition, so the full un-tiled fft-3way kernel is run once
// per dModel-slice (dM = dModel/nb_d channels). Each slice's intermediates use the
// proven un-tiled TCDM sum layout (separate buffers, matching bank offsets); the
// input slice is DMA'd in, the partition3 output is scattered into the full L3
// buffer. See docs/dataflow/05_fft.md §5.4.

#include "../data/data.h"
#include "snax-simbacore-lib.h"

static inline uint32_t align64(uint32_t x) { return (x + 63u) & ~63u; }

int test() {
    int err = 0;

    // L3: reserve 16 KiB past the runtime's putc_buffer, then a FULL partition3_out
    // output buffer that the per-slice scatter assembles into.
    static uint8_t* ptr_output_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_output_l3 = (uint8_t*)snrt_l3alloc(M6_length_partition3_out_full);
    }
    snrt_cluster_hw_barrier();

    // TCDM: un-tiled fft-3way sum layout (weights/twiddles FULL, intermediates dM-sized).
    void* base                   = snrt_l1_next();
    uint8_t* ptr_weight1         = (uint8_t*)base + M6_addr_weight1;
    uint8_t* ptr_weight2         = (uint8_t*)base + M6_addr_weight2;
    uint8_t* ptr_weight3         = (uint8_t*)base + M6_addr_weight3;
    uint8_t* ptr_in              = (uint8_t*)base + M6_addr_in;
    uint8_t* ptr_partition1_out  = (uint8_t*)base + M6_addr_partition1_out;
    uint8_t* ptr_twiddles1       = (uint8_t*)base + M6_addr_twiddles1;
    uint8_t* ptr_hadamard1_out   = (uint8_t*)base + M6_addr_hadamard1_out;
    uint8_t* ptr_hadamard1_packed = (uint8_t*)base + M6_addr_hadamard1_packed;
    uint8_t* ptr_partition2_out  = (uint8_t*)base + M6_addr_partition2_out;
    uint8_t* ptr_twiddles2       = (uint8_t*)base + M6_addr_twiddles2;
    uint8_t* ptr_hadamard2_out   = (uint8_t*)base + M6_addr_hadamard2_out;
    uint8_t* ptr_hadamard2_packed = (uint8_t*)base + M6_addr_hadamard2_packed;
    uint8_t* ptr_partition3_out  = (uint8_t*)base + M6_addr_partition3_out;

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Weights and twiddles: loaded once (depend only on L1/L2/L3, broadcast over d).
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

    uint32_t nblk = (2 * L3) / seqLenUnroll;  // partition3 output row-blocks (M_3)

    for (uint32_t s = 0; s < M6_nb_d; s++) {
        // Load this slice's input; zero the three partition psum buffers.
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_in, M6_dft_in + s * M6_length_in_slice, M6_length_in_slice);
            snrt_dma_start_1d(ptr_partition1_out, (void*)snrt_zero_memory_ptr(), M6_length_partition1_out);
            snrt_dma_start_1d(ptr_partition2_out, (void*)snrt_zero_memory_ptr(), M6_length_partition2_out);
            snrt_dma_start_1d(ptr_partition3_out, (void*)snrt_zero_memory_ptr(), M6_length_partition3_out);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            // Step 1: partition 1.
            set_isgemm_streamer_csr((uint32_t)ptr_weight1, M6_R11_1_ss, M6_R11_1_tb, M6_R11_1_ts,
                                    (uint32_t)ptr_in, M6_R12_1_ss, M6_R12_1_tb, M6_R12_1_ts,
                                    (uint32_t)ptr_partition1_out, M6_W3_1_ss, M6_W3_1_tb, M6_W3_1_ts);
            set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L1, 1, L1_padded, 1, M6_dModel_slice * L2 * L3);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Step 2: hadamard 1 (CMUL FP8).
            set_simd_streamer_csr((uint32_t)ptr_partition1_out, M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts,
                                  (uint32_t)ptr_twiddles1, M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts,
                                  (uint32_t)ptr_hadamard1_out, M6_W3_2_ss, M6_W3_2_tb, M6_W3_2_ts);
            set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Step 2B: reorder 1 (SIMD NOOP deinterleave).
            set_simd_streamer_no_b((uint32_t)ptr_hadamard1_out, M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts,
                                   (uint32_t)ptr_hadamard1_packed, M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts);
            set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Step 3: partition 2.
            set_isgemm_streamer_csr((uint32_t)ptr_weight2, M6_R11_3_ss, M6_R11_3_tb, M6_R11_3_ts,
                                    (uint32_t)ptr_hadamard1_packed, M6_R12_3_ss, M6_R12_3_tb, M6_R12_3_ts,
                                    (uint32_t)ptr_partition2_out, M6_W3_3_ss, M6_W3_3_tb, M6_W3_3_ts);
            set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L2, 1, 2 * L2_padded, 1, M6_dModel_slice * L1 * L3);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Step 4: hadamard 2 (CMUL FP8).
            set_simd_streamer_csr((uint32_t)ptr_partition2_out, M6_R7_4_ss, M6_R7_4_tb, M6_R7_4_ts,
                                  (uint32_t)ptr_twiddles2, M6_R13_4_ss, M6_R13_4_tb, M6_R13_4_ts,
                                  (uint32_t)ptr_hadamard2_out, M6_W3_4_ss, M6_W3_4_tb, M6_W3_4_ts);
            set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Step 4B: reorder 2 (SIMD NOOP deinterleave).
            set_simd_streamer_no_b((uint32_t)ptr_hadamard2_out, M6_R7_4B_ss, M6_R7_4B_tb, M6_R7_4B_ts,
                                   (uint32_t)ptr_hadamard2_packed, M6_W3_4B_ss, M6_W3_4B_tb, M6_W3_4B_ts);
            set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Step 5: partition 3.
            set_isgemm_streamer_csr((uint32_t)ptr_weight3, M6_R11_5_ss, M6_R11_5_tb, M6_R11_5_ts,
                                    (uint32_t)ptr_hadamard2_packed, M6_R12_5_ss, M6_R12_5_tb, M6_R12_5_ts,
                                    (uint32_t)ptr_partition3_out, M6_W3_5_ss, M6_W3_5_tb, M6_W3_5_ts);
            set_simbacore_csr(M6_ISGEMM_SQ, 2 * L3, 1, 2 * L3_padded, 1, M6_dModel_slice * L1 * L2);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            asm volatile("fence" ::: "memory");
        }
        snrt_cluster_hw_barrier();

        // Scatter this slice's partition3_out into its d-slice of the full L3 output.
        // The output is flattened K_M_N (d outer, Mu = seqLenUnroll), so a d-slice is
        // M_3 = 2*L3/seqLenUnroll strided blocks, not one contiguous chunk.
        if (snrt_is_dm_core()) {
            uint32_t slice_block = M6_length_partition3_out_slice / nblk;
            uint32_t full_block  = M6_length_partition3_out_full / nblk;
            snrt_dma_start_2d(ptr_output_l3 + s * slice_block, ptr_partition3_out, slice_block,
                              full_block, slice_block, nblk);
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
