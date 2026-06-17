// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled 3-way partitioned EinFFT (L = L1 * L2 * L3): the un-tiled kernel run
// once per dModel-slice, with the DM core's DMA hidden behind compute.
// Dataflow (per-slice kernel, buffer sizing, latency hiding):
// docs/dataflow/05_fft.md, "fft-3way-tiled" section.

#include "../data/data.h"
#include "snax-simbacore-lib.h"

static inline uint32_t align64(uint32_t x) { return (x + 63u) & ~63u; }

static inline void dma_load_input_slice(uint8_t* dst, uint32_t s) {
    snrt_dma_start_2d(dst, M6_dft_in + s * M6_in_slice_chunk, M6_in_slice_chunk,
                      /*dst_stride=*/M6_in_slice_chunk, /*src_stride=*/M6_in_ktile_stride,
                      /*repeat=*/M6_in_ktile_count);
}

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

    // TCDM layout. Weights/twiddles depend only on L (broadcast over d) → live.
    // Then dedicated per-slice buffers instead of the old 2 max-sized slots:
    //   in : FP8 slice input (gemm1 R12)
    //   P  : the BF16 partition psum (slot_size) reused by gemm1/2/3 in turn
    //   H1 : FP8 CMul output (slot_size/2)
    //   H2 : FP8 reorder output (slot_size/2)
    // Right-sizing H1/H2 to FP8 frees the room the DM core's background work needs.
    const uint32_t hsize   = M6_slot_size / 2;  // FP8 hadamard buffers = half a BF16 slot
    void* tcdm_base_ptr    = snrt_l1_next();
    uint8_t* ptr_weight1   = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_weight2   = ptr_weight1 + align64(M6_length_weight1);
    uint8_t* ptr_weight3   = ptr_weight2 + align64(M6_length_weight2);
    uint8_t* ptr_twiddles1 = ptr_weight3 + align64(M6_length_weight3);
    uint8_t* ptr_twiddles2 = ptr_twiddles1 + align64(M6_length_twiddles1);
    uint8_t* ptr_in        = ptr_twiddles2 + align64(M6_length_twiddles2);
    uint8_t* ptr_P         = ptr_in + align64(M6_length_in_slice);
    uint8_t* ptr_H1        = ptr_P + M6_slot_size;
    uint8_t* ptr_H2        = ptr_H1 + hsize;

    uint32_t start_cycles     = 0;
    uint32_t simbacore_cycles = 0;

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: tiled 3-way FFT (outer dModel-tile, nb_d=%d, seqLen=%d, dModel=%d, "
            "L1=%d, L2=%d, L3=%d)\n\n",
            M6_nb_d, seqLen, dModel, L1, L2, L3);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
    }

    // Weights and twiddles: loaded once, broadcast across all dModel slices.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_weight3, M6_dft_weight3, M6_length_weight3);
        snrt_dma_start_1d(ptr_twiddles1, M6_twiddles1, M6_length_twiddles1);
        snrt_dma_start_1d(ptr_twiddles2, M6_twiddles2, M6_length_twiddles2);
        // Prologue: slice 0 input + zero the gemm1 psum (rest is prefetched per slice).
        dma_load_input_slice(ptr_in, 0);
        snrt_dma_start_1d(ptr_P, (void*)snrt_zero_memory_ptr(), M6_slot_size);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // Software pipeline: the streamer latches its CSRs at start, so each step's
    // streamer config is programmed *while the previous step still runs* — hiding
    // the ~50 config writes/step behind the accelerator. The 6-write simbacore
    // MODE CSR stays serial (issued right before each start) to avoid retuning the
    // running core's mode. Helper macros keep the interleave readable.
#define CFG_GEMM(W, IN_, N)                                                                                     \
    set_isgemm_streamer_csr((uint32_t)(W), M6_R11_##N##_ss, M6_R11_##N##_tb, M6_R11_##N##_ts, (uint32_t)(IN_),  \
                            M6_R12_##N##_ss, M6_R12_##N##_tb, M6_R12_##N##_ts, (uint32_t)ptr_P, M6_W3_##N##_ss, \
                            M6_W3_##N##_tb, M6_W3_##N##_ts)
    for (uint32_t s = 0; s < M6_nb_d; s++) {
        if (snrt_global_core_idx() == 0) {
            // step 1 (gemm1, in→P): config serial, then start; overlap cmul1 config.
            CFG_GEMM(ptr_weight1, ptr_in, 1);
            set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L1, 1, L1_padded, 1, M6_dModel_slice * L2 * L3);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            set_simd_streamer_csr((uint32_t)ptr_P, M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts, (uint32_t)ptr_twiddles1,
                                  M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts, (uint32_t)ptr_H1, M6_W3_2_ss, M6_W3_2_tb,
                                  M6_W3_2_ts);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // step 2 (cmul1, P→H1): start; overlap noop1 config.
            set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            set_simd_streamer_no_b((uint32_t)ptr_H1, M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts, (uint32_t)ptr_H2,
                                   M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();  // A: P free (cmul1 read it), H1 ready

        // DM zeros P for gemm2 and prefetches next slice's input, hidden behind reorder 1.
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_P, (void*)snrt_zero_memory_ptr(), M6_slot_size);
            if (s + 1 < M6_nb_d) dma_load_input_slice(ptr_in, s + 1);
            snrt_dma_wait_all();
        }
        if (snrt_global_core_idx() == 0) {
            // step 2B (noop1, H1→H2): streamer cfg done above; start; overlap gemm2 config.
            set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            CFG_GEMM(ptr_weight2, ptr_H2, 3);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();  // B: P zeroed, H2 ready, next input prefetched

        if (snrt_global_core_idx() == 0) {
            // step 3 (gemm2, H2→P): streamer cfg done above; start; overlap cmul2 config.
            set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L2, 1, 2 * L2_padded, 1, M6_dModel_slice * L1 * L3);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            set_simd_streamer_csr((uint32_t)ptr_P, M6_R7_4_ss, M6_R7_4_tb, M6_R7_4_ts, (uint32_t)ptr_twiddles2,
                                  M6_R13_4_ss, M6_R13_4_tb, M6_R13_4_ts, (uint32_t)ptr_H1, M6_W3_4_ss, M6_W3_4_tb,
                                  M6_W3_4_ts);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // step 4 (cmul2, P→H1): start; overlap noop2 config.
            set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            set_simd_streamer_no_b((uint32_t)ptr_H1, M6_R7_4B_ss, M6_R7_4B_tb, M6_R7_4B_ts, (uint32_t)ptr_H2,
                                   M6_W3_4B_ss, M6_W3_4B_tb, M6_W3_4B_ts);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();  // C: P free (cmul2 read it), H1 ready

        // DM zeros P for gemm3, hidden behind reorder 2.
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_P, (void*)snrt_zero_memory_ptr(), M6_slot_size);
            snrt_dma_wait_all();
        }
        if (snrt_global_core_idx() == 0) {
            // step 4B (noop2, H1→H2): streamer cfg done above; start; overlap gemm3 config.
            set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            CFG_GEMM(ptr_weight3, ptr_H2, 5);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();  // D: P zeroed, H2 ready

        if (snrt_global_core_idx() == 0) {
            // step 5 (gemm3, H2→P): streamer cfg done above; start.
            set_simbacore_csr(M6_ISGEMM_SQ, 2 * L3, 1, 2 * L3_padded, 1, M6_dModel_slice * L1 * L2);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // Ensure P is drained to TCDM before the DMA-out reads it.
            asm volatile("fence" ::: "memory");
        }
        snrt_cluster_hw_barrier();  // E: partition3_out (P) ready

        // Scatter this slice's output into its d-slice of the full L3 output, then
        // re-zero P for the next slice's gemm1. See docs/dataflow/05_fft.md §5.4.
        if (snrt_is_dm_core()) {
            snrt_dma_start_2d(ptr_output_l3 + s * M6_out_block_slice, ptr_P, M6_out_block_slice,
                              /*dst_stride=*/M6_out_block_full, /*src_stride=*/M6_out_block_slice,
                              /*repeat=*/M6_out_nblk);
            snrt_dma_wait_all();
            if (s + 1 < M6_nb_d) {
                snrt_dma_start_1d(ptr_P, (void*)snrt_zero_memory_ptr(), M6_slot_size);
                snrt_dma_wait_all();
            }
        }
        snrt_cluster_hw_barrier();  // F: scattered + P zeroed for next slice
    }
#undef CFG_GEMM

    // --- Verification: full output assembled in L3, scalar-read at sample positions. ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_output_l3, M6_partition3_expected, M6_test_samples_expected, nb_test_samples,
                                   "partition3_out (L3)");

        printf("Test FFT 3-way tiled (outer dModel-tile): (%d x %d), L1=%d L2=%d L3=%d, nb_d=%d\n", seqLen, dModel, L1,
               L2, L3, M6_nb_d);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
