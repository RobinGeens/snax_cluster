// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Standalone folded-BatchNorm + ReLU: out = ReLU(x * scale + shift), with per-channel
// scale/shift. This is the SegFormer ConvModule tail that follows the 1x1-conv (= OS-core
// GEMM, the existing osgemm app). A single fused SIMD FmaHold+ReLU pass: the b-operand
// stream carries one (scale, shift) pair per channel, held for the channel's L/lanes outputs.

#include <stdint.h>

#include "../data/data.h"
#include "snax-simbacore-lib.h"

// start/wait pair: the SIMD passes never use the R10/R11 delayed streamers, so their delay-gauge
// polling and clears are skipped (perf doc docs/dataflow/08_performance_optimization.md section 3).
// START is a pulse -- the streamer latches its config at the rising edge.
static inline void simd_start(uint32_t mode) {
    write_csr(MODE, mode);  // MODE is read in real time, must be set before START (never preloaded)
    _set_streamer_start();
    _set_simbacore_start();
    write_csr(STREAMER_START_CSR, 0);
    write_csr(SIMBACORE_START, 0);
}
static inline void simd_wait(void) {
    while (read_csr(SIMBACORE_BUSY));
    while (read_csr(STREAMER_BUSY_CSR));
}

int test() {
    int err = 0;

    void* tcdm_base_ptr      = snrt_l1_next();
    uint16_t* ptr_x          = (uint16_t*)(tcdm_base_ptr + M10_addr_x);
    uint16_t* ptr_scaleshift = (uint16_t*)(tcdm_base_ptr + M10_addr_scaleshift);
    uint16_t* ptr_out        = (uint16_t*)(tcdm_base_ptr + M10_addr_out);

    uint32_t start_cycles = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: BatchNorm + ReLU (%u x %u)\n\n", seqLen, channels);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();

        // Program pass-1's streamer config now so its ~45 CSR writes overlap the DM core's load
        // DMA below (CSR writes touch only streamer config, not TCDM) instead of sitting on the
        // critical path after the barrier.
        set_simd_streamer_csr((uint32_t)ptr_x, M10_R7_xw_ss, M10_R7_xw_tb, M10_R7_xw_ts,                 //
                              (uint32_t)ptr_scaleshift, M10_R13_vec_ss, M10_R13_vec_tb, M10_R13_vec_ts,  //
                              (uint32_t)ptr_out, M10_W3_xw_ss, M10_W3_xw_tb, M10_W3_xw_ts);
    }

    // Load activations + interleaved per-channel scale/shift from L3.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_x, M10_x, M10_length_x);
        snrt_dma_start_1d(ptr_scaleshift, M10_scaleshift, M10_length_scaleshift);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        // Single fused pass: out = ReLU(x * scale + shift). The R13 stream delivers one
        // interleaved (scale, shift) pair per channel; n_acc = outputs per pair.
        set_simbacore_simd_n_acc(seqLen / simdLanes_bf16);
        simd_start(M48_SIMD_FMA_HOLD_BF16_RELU);
        simd_wait();

        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%u cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample_u16(ptr_out, M10_out, M10_test_samples_out, nb_test_samples, "out");

        printf("Test BatchNorm + ReLU: (%u x %u)\n", seqLen, channels);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
