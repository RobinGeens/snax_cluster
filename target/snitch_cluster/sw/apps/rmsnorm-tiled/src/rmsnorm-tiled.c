// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// RMSNorm plus residual, tiled in seqLen (out = rmsnorm(x)*weight + y)
// See docs/dataflow/14_rmsnorm_tiled.md.
// Six SIMD passes: the five-pass RMSNorm (Sqrt/Div fold via the sqrt(D) plus one big Add for the residual
//   1. Rms : rms <- Sum(x^2)             (big: LtD)
//   2. Sqrt: rms <- sqrt(rms)            (small: Lt) = sqrt(Sum x^2)
//   3. Div : rms <- sqrt(D) / rms        (small: Lt) = sqrt(D)/sqrt(Sum x^2) = 1/rms_true
//   4. Mul : x   <- x * rms              (big: 1/rms_true broadcast over D)
//   5. Mul : x   <- x * weight           (big: per-channel weight)
//   6. Add : x   <- x + y                (big: elementwise residual)
// Net: x * sqrt(D)/sqrt(Sum x^2) * weight + y = x / rms_true * weight + y.

#include <math.h>
#include <stdint.h>

#include "../data/data.h"
#include "snax-simbacore-lib.h"

/* BF16 = high 16 bits of IEEE 754 float (truncation, no rounding). */
static inline uint16_t fp32_to_bf16(float f) {
    union {
        float f;
        uint32_t u;
    } x = {.f = f};
    return (uint16_t)(x.u >> 16);
}

// Temporal strides for broadcast
static const int32_t zero_ts[4] = {0, 0, 0, 0};

// Launch the SIMD core + streamers for one pass. Inlined instead of using the library
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

static inline void normalize_tile(uint16_t* ptr_x, uint16_t* ptr_y, uint16_t* ptr_rms, uint16_t* ptr_weight,
                                  uint16_t* ptr_sqrtd) {
    // Six SIMD passes (header). After each START the *next* pass's streamer program is written while
    // the current pass computes -- the streamer latches config at the next START, so the ~48 per-port
    // writes hide behind compute (perf doc 08 section 1: CSR preload). The full per-port re-program
    // is kept every pass: a lean program leaves a streamer mis-armed.

    // 1. rms = Sum(x^2) per row
    set_simd_streamer_no_b((uint32_t)ptr_x, M12_R7_x_ss, M12_R7_x_tb, M12_R7_x_ts,  //
                           (uint32_t)ptr_rms, M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);
    simd_start(M13_SIMD_RMS_BF16);
    set_simd_streamer_no_b((uint32_t)ptr_rms, M12_R7_rms_ss, M12_R7_rms_tb, M12_R7_rms_ts,  // preload pass 2
                           (uint32_t)ptr_rms, M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);
    simd_wait();

    // 2. rms = sqrt(rms)
    simd_start(M15_SIMD_SQRT_BF16);
    set_simd_streamer_csr((uint32_t)ptr_sqrtd, M12_R7_rms_ss, M12_R7_rms_tb, (int32_t*)zero_ts,  // preload pass 3
                          (uint32_t)ptr_rms, M12_R7_rms_ss, M12_R7_rms_tb, M12_R7_rms_ts,        //
                          (uint32_t)ptr_rms, M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);
    simd_wait();

    // 3. rms = sqrt(D)/rms = 1/rms_true   (sqrt(D) numerator folds in the "* 1/D" of the true rms)
    simd_start(M14_SIMD_DIV_BF16);
    set_simd_streamer_csr((uint32_t)ptr_x, M12_R7_x_ss, M12_R7_x_tb, M12_R7_x_ts,                    // preload pass 4
                          (uint32_t)ptr_rms, M12_R13_x_rms_ss, M12_R13_x_rms_tb, M12_R13_x_rms_ts,   //
                          (uint32_t)ptr_x, M12_W3_x_ss, M12_W3_x_tb, M12_W3_x_ts);
    simd_wait();

    // 4. x = x * invrms   (1/rms broadcast over D, one per token)
    simd_start(M10_SIMD_MUL_BF16);
    set_simd_streamer_csr((uint32_t)ptr_x, M12_R7_x_w_ss, M12_R7_x_w_tb, M12_R7_x_w_ts,           // preload pass 5
                          (uint32_t)ptr_weight, M12_R13_x_w_ss, M12_R13_x_w_tb, M12_R13_x_w_ts,   //
                          (uint32_t)ptr_x, M12_W3_x_w_ss, M12_W3_x_w_tb, M12_W3_x_w_ts);
    simd_wait();

    // 5. x = x * weight   (per channel, stationary over L)
    simd_start(M10_SIMD_MUL_BF16);
    set_simd_streamer_csr((uint32_t)ptr_x, M12_R7_x_ss, M12_R7_x_tb, M12_R7_x_ts,      // preload pass 6
                          (uint32_t)ptr_y, M12_R13_y_ss, M12_R13_y_tb, M12_R13_y_ts,   //
                          (uint32_t)ptr_x, M12_W3_x_ss, M12_W3_x_tb, M12_W3_x_ts);
    simd_wait();

    // 6. x = x + y   (elementwise residual; y full-tile via R13)
    simd_start(M8_SIMD_ADD_BF16);
    simd_wait();

    // Drain the SIMD write buffer to TCDM before the DM core DMAs this slot out.
    asm volatile("fence" ::: "memory");
}

int test() {
    int err = 0;

    // TCDM layout: resident weight + sqrt(D) constant + one rms scratch + two x slots + two y slots.
    void* base           = snrt_l1_next();
    uint16_t* ptr_weight = (uint16_t*)base;
    uint16_t* ptr_sqrtd  = (uint16_t*)((uint8_t*)ptr_weight + M12_length_weight);
    uint16_t* ptr_rms    = (uint16_t*)((uint8_t*)ptr_sqrtd + simdLanes_bf16 * 2);
    uint16_t* ptr_x[2];
    ptr_x[0] = (uint16_t*)((uint8_t*)ptr_rms + M12_length_rms_tile);
    ptr_x[1] = (uint16_t*)((uint8_t*)ptr_x[0] + M12_length_x_tile);
    uint16_t* ptr_y[2];
    ptr_y[0] = (uint16_t*)((uint8_t*)ptr_x[1] + M12_length_x_tile);
    ptr_y[1] = (uint16_t*)((uint8_t*)ptr_y[0] + M12_length_x_tile);

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;

    // L3 staging buffer for the normalized output (full seqLen x dModel).
    static uint8_t* l3_out;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: RMSNorm+residual tiled (seqLen=%u dModel=%u nb_tiles=%u L_tile=%u)\n\n", seqLen,
               dModel, nb_tiles, M12_L_tile);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();

        // Reserve 16 KiB so snrt_l3alloc does not overlap the runtime's putc_buffer.
        (void)snrt_l3alloc(16 * 1024);
        l3_out = (uint8_t*)snrt_l3alloc((uint32_t)nb_tiles * M12_length_x_tile);
    }

    // Load resident weight + first x and y tiles.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d((uint8_t*)ptr_weight, (uint8_t*)M12_weight, M12_length_weight);
        snrt_dma_start_1d((uint8_t*)ptr_x[0], (uint8_t*)M12_x, M12_length_x_tile);
        snrt_dma_start_1d((uint8_t*)ptr_y[0], (uint8_t*)M12_y, M12_length_x_tile);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        // Fill the Div-numerator lane constant with sqrt(D): folds the "* 1/D" of the true rms into
        // the reciprocal pass, removing one SIMD pass per tile at no per-tile cost.
        uint16_t sqrtd = fp32_to_bf16(sqrtf((float)dModel));
        for (uint32_t i = 0; i < simdLanes_bf16; i++) ptr_sqrtd[i] = sqrtd;

        set_simbacore_simd_n_acc(dModel);  // Rms reduction length, set once
    }
    snrt_cluster_hw_barrier();

    // Double-buffered pipeline. While the SIMD core normalizes tile n (slot n&1), the DM core
    // (concurrently) DMAs tile n-1's result out and tile n+1's input in (both via slot (n+1)&1).
    for (uint32_t n = 0; n < nb_tiles; n++) {
        uint32_t s     = n & 1;
        uint32_t other = s ^ 1;

        if (snrt_is_dm_core()) {
            if (n >= 1)
                snrt_dma_start_1d(l3_out + (n - 1) * M12_length_x_tile, (uint8_t*)ptr_x[other], M12_length_x_tile);
            if (n + 1 < nb_tiles) {
                snrt_dma_start_1d((uint8_t*)ptr_x[other], (uint8_t*)M12_x + (n + 1) * M12_length_x_tile,
                                  M12_length_x_tile);
                snrt_dma_start_1d((uint8_t*)ptr_y[other], (uint8_t*)M12_y + (n + 1) * M12_length_x_tile,
                                  M12_length_x_tile);
            }
        }

        if (snrt_global_core_idx() == 0) {
            normalize_tile(ptr_x[s], ptr_y[s], ptr_rms, ptr_weight, ptr_sqrtd);
            simbacore_cycles_total += read_simbacore_perf_counter();
        }

        snrt_cluster_hw_barrier();                   // compute tile n done
        if (snrt_is_dm_core()) snrt_dma_wait_all();  // tile n+1 input ready, tile n-1 output drained
        snrt_cluster_hw_barrier();
    }

    // Flush the last tile's output (issued after the loop since it has no successor iteration).
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l3_out + (nb_tiles - 1) * M12_length_x_tile, (uint8_t*)ptr_x[(nb_tiles - 1) & 1],
                          M12_length_x_tile);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_total);
        printf("[%u cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample_u16((uint16_t*)l3_out, M12_out, M12_test_samples_expected, nb_test_samples, "out");

        printf("Test RMSNorm+residual tiled: (%u x %u), nb_tiles=%u\n", seqLen, dModel, nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
