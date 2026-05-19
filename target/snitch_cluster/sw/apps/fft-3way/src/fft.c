// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// 3-way partitioned EinFFT (L = L1 * L2 * L3). See data/datagen.py for pipeline overview.
//
// Pipeline (fully streamer-driven, no software byte-loops between stages):
//   1.  partition1  IS-core (ISGEMM_SQ_TRANSPOSE) of W_L1 · stack_L1
//   2.  hadamard1   SIMD CMUL_FP8 with twiddles1 broadcast across dModel
//   2B. reorder1    SIMD NOOP_FP8 — deinterleaves re/im into __flattenColMajor
//   3.  partition2  IS-core (ISGEMM_SQ_TRANSPOSE) reads hadamard1_packed via R12
//   4.  hadamard2   SIMD CMUL_FP8 with twiddles2 broadcast across L1 and dModel
//   4B. reorder2    SIMD NOOP_FP8 — deinterleaves re/im
//   5.  partition3  IS-core (ISGEMM_SQ) reads hadamard2_packed via R12
//
// Why no inter-stage byte reorders are needed:
//   The Scala data generator emits `dft_in` / `twiddles1` / `partition1_expected`
//   / `hadamard1_expected` in the chip's "NEW" byte layout where the inner-
//   (L2*L3) index is `col = d*L2*L3 + m3*L2 + m2` instead of the standard
//   `m2*L3 + m3`. Under that layout, the SIMD-NOOP deinterleave produces a
//   `hadamard1_packed` whose 16-byte chunks are 16 stride-1 m2 values for one
//   (k1, m3, d) cell — exactly partition2's K-tile format. Partition2's R12
//   walks chunks linearly with the col arrangement `d*L1*L3 + k1*L3 + m3`
//   (see `buildStackedInput_L2_3way` in EinfftLib.scala).
//
//   The same trick is used after partition2: `buildStackedInput_L3_3way`'s
//   `col = d*L1*L2 + k1*L2 + k2` layout matches the natural chunk order of
//   the deinterleaved post-stage-2 buffer, so partition3 reads it directly.

#include "../data/data.h"
#include "snax-simbacore-lib.h"

int test() {
    int err = 0;

    // ----- TCDM address layout -------------------------------------------------
    void* tcdm_base_ptr            = snrt_l1_next();
    uint8_t* ptr_weight1           = (uint8_t*)(tcdm_base_ptr + M6_addr_weight1);
    uint8_t* ptr_weight2           = (uint8_t*)(tcdm_base_ptr + M6_addr_weight2);
    uint8_t* ptr_weight3           = (uint8_t*)(tcdm_base_ptr + M6_addr_weight3);
    uint8_t* ptr_in                = (uint8_t*)(tcdm_base_ptr + M6_addr_in);
    uint8_t* ptr_partition1_out    = (uint8_t*)(tcdm_base_ptr + M6_addr_partition1_out);
    uint8_t* ptr_twiddles1         = (uint8_t*)(tcdm_base_ptr + M6_addr_twiddles1);
    uint8_t* ptr_hadamard1_out     = (uint8_t*)(tcdm_base_ptr + M6_addr_hadamard1_out);
    // `hadamard1_packed` is partition2's input directly — chip-side NEW byte layout
    // (m2 stride-1 in 16-byte chunks) is produced by the SIMD-NOOP deinterleave.
    uint8_t* ptr_hadamard1_packed  = (uint8_t*)(tcdm_base_ptr + M6_addr_hadamard1_packed);
    uint8_t* ptr_partition2_out    = (uint8_t*)(tcdm_base_ptr + M6_addr_partition2_out);
    uint8_t* ptr_twiddles2         = (uint8_t*)(tcdm_base_ptr + M6_addr_twiddles2);
    uint8_t* ptr_hadamard2_out     = (uint8_t*)(tcdm_base_ptr + M6_addr_hadamard2_out);
    uint8_t* ptr_hadamard2_packed  = (uint8_t*)(tcdm_base_ptr + M6_addr_hadamard2_packed);
    uint8_t* ptr_partition3_out    = (uint8_t*)(tcdm_base_ptr + M6_addr_partition3_out);

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // ----- DMA-in all inputs and zero psum buffers -----------------------------
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_weight3, M6_dft_weight3, M6_length_weight3);
        snrt_dma_start_1d(ptr_in, M6_dft_in, M6_length_in);
        snrt_dma_start_1d(ptr_twiddles1, M6_twiddles1, M6_length_twiddles1);
        snrt_dma_start_1d(ptr_twiddles2, M6_twiddles2, M6_length_twiddles2);
        snrt_dma_start_1d(ptr_partition1_out, (void*)snrt_zero_memory_ptr(), M6_length_partition1_out);
        snrt_dma_start_1d(ptr_partition2_out, (void*)snrt_zero_memory_ptr(), M6_length_partition2_out);
        snrt_dma_start_1d(ptr_partition3_out, (void*)snrt_zero_memory_ptr(), M6_length_partition3_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles = 0;
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: 3-way partitioned FFT (L=%d, dModel=%d, L1=%d, L2=%d, L3=%d)\n\n",
               seqLen, dModel, L1, L2, L3);
        start_cycles = snrt_mcycle();

        // ===== Step 1: partition 1 =========================================
        set_isgemm_streamer_csr((uint32_t)ptr_weight1, M6_R11_1_ss, M6_R11_1_tb, M6_R11_1_ts,         // A
                                (uint32_t)ptr_in, M6_R12_1_ss, M6_R12_1_tb, M6_R12_1_ts,              // B
                                (uint32_t)ptr_partition1_out, M6_W3_1_ss, M6_W3_1_tb, M6_W3_1_ts);    // C/D
        set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L1, 1, L1_padded, 1, (dModel * L2 * L3));
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();

        // ===== Step 2: hadamard 1 (CMUL FP8) ==============================
        set_simd_streamer_csr((uint32_t)ptr_partition1_out, M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts,
                              (uint32_t)ptr_twiddles1, M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts,
                              (uint32_t)ptr_hadamard1_out, M6_W3_2_ss, M6_W3_2_tb, M6_W3_2_ts);
        set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // ===== Step 2B: deinterleave re/im (SIMD NOOP) ====================
        set_simd_streamer_no_b((uint32_t)ptr_hadamard1_out, M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts,
                               (uint32_t)ptr_hadamard1_packed, M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts);
        set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
    }
    if (snrt_global_core_idx() == 0) {
        // ===== Step 3: partition 2 ====================================
        // ISGEMM_SQ_TRANSPOSE: the IS-core output goes through the bank transposer
        // (FP8, banked), so the downstream SIMD CMul step 4 can read it via the
        // banked port. Reads `hadamard1_packed` directly — the NEW byte layout
        // (m2 stride-1 in 16-byte chunks) is the partition2 B input format.
        set_isgemm_streamer_csr((uint32_t)ptr_weight2, M6_R11_3_ss, M6_R11_3_tb, M6_R11_3_ts,
                                (uint32_t)ptr_hadamard1_packed, M6_R12_3_ss, M6_R12_3_tb, M6_R12_3_ts,
                                (uint32_t)ptr_partition2_out, M6_W3_3_ss, M6_W3_3_tb, M6_W3_3_ts);
        set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L2, 1, 2 * L2_padded, 1, (dModel * L1 * L3));
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();

        // ===== Step 4: hadamard 2 (CMUL FP8) ==========================
        set_simd_streamer_csr((uint32_t)ptr_partition2_out, M6_R7_4_ss, M6_R7_4_tb, M6_R7_4_ts,
                              (uint32_t)ptr_twiddles2, M6_R13_4_ss, M6_R13_4_tb, M6_R13_4_ts,
                              (uint32_t)ptr_hadamard2_out, M6_W3_4_ss, M6_W3_4_tb, M6_W3_4_ts);
        set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // ===== Step 4B: deinterleave re/im (SIMD NOOP) ================
        set_simd_streamer_no_b((uint32_t)ptr_hadamard2_out, M6_R7_4B_ss, M6_R7_4B_tb, M6_R7_4B_ts,
                               (uint32_t)ptr_hadamard2_packed, M6_W3_4B_ss, M6_W3_4B_tb, M6_W3_4B_ts);
        set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        // ===== Step 5: partition 3 ====================================
        // Reads `hadamard2_packed` directly: with `buildStackedInput_L3_3way`'s col
        // layout (d outer, k1 middle, k2 inner) matching the natural chunk order of
        // the SIMD-NOOP deinterleave output, no inter-stage reorder is needed.
        set_isgemm_streamer_csr((uint32_t)ptr_weight3, M6_R11_5_ss, M6_R11_5_tb, M6_R11_5_ts,
                                (uint32_t)ptr_hadamard2_packed, M6_R12_5_ss, M6_R12_5_tb, M6_R12_5_ts,
                                (uint32_t)ptr_partition3_out, M6_W3_5_ss, M6_W3_5_tb, M6_W3_5_ts);
        set_simbacore_csr(M6_ISGEMM_SQ, 2 * L3, 1, 2 * L3_padded, 1, (dModel * L1 * L2));
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();

        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        // ----- Verification --------------------------------------------------
        err += check_result_sample(ptr_partition1_out, M6_partition1_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition 1");
        err += check_result_sample((uint8_t*)ptr_hadamard1_out, M6_hadamard1_expected, M6_test_samples_expected,
                                   nb_test_samples, "hadamard 1");
        err += check_result_sample(ptr_partition2_out, M6_partition2_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition 2");
        err += check_result_sample((uint8_t*)ptr_hadamard2_out, M6_hadamard2_expected, M6_test_samples_expected,
                                   nb_test_samples, "hadamard 2");
        err += check_result_sample(ptr_partition3_out, M6_partition3_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition 3");

        printf("Test FFT 3-way: (%d x %d), L1=%d L2=%d L3=%d\n", seqLen, dModel, L1, L2, L3);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 5 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
