// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// RMSNorm tiled in seqLen, optimized to minimize SimbaCore/streamer launch overhead.
// See docs/dataflow/14_rmsnorm_tiled.md.

#include <stdint.h>
#include <math.h>

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

// Temporal strides for broadcast (constant not advanced): all zeros
static const int32_t zero_ts[4] = {0, 0, 0, 0};

// Normalize one L-tile (Lt rows) in place in `ptr_x`, using `ptr_rms` (Lt elems) as scratch.
//
// Five SIMD passes instead of the six in the non-tiled `rmsnorm` app: the "* 1/D" pass is folded
// into the reciprocal pass by using sqrt(D) (not 1.0) as the Div numerator -- a free 16-lane
// constant, no extra work. The expensive Sqrt/Div run on the tiny per-token rms vector (Lt elems)
// only -- never on the full x tile -- because they wrap an iterative unit ~10x slower per element;
// the full-tile work stays cheap Rms/Mul:
//   1. Rms : rms <- Sum(x^2)             (big: LtD)
//   2. Sqrt: rms <- sqrt(rms)            (small: Lt) = sqrt(Sum x^2)
//   3. Div : rms <- sqrt(D) / rms        (small: Lt) = sqrt(D)/sqrt(Sum x^2) = 1/rms_true
//   4. Mul : x   <- x * rms              (big: 1/rms_true broadcast over D)
//   5. Mul : x   <- x * weight           (big: per-channel weight)
// Net: x * sqrt(D)/sqrt(Sum x^2) * weight = x / sqrt(mean x^2) * weight = x / rms_true * weight.
static inline void normalize_tile(uint16_t* ptr_x, uint16_t* ptr_rms, uint16_t* ptr_weight, uint16_t* ptr_sqrtd) {
    // 1. rms = Sum(x^2) per row
    set_simd_streamer_no_b((uint32_t)ptr_x, M12_R7_x_ss, M12_R7_x_tb, M12_R7_x_ts,  //
                           (uint32_t)ptr_rms, M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);
    set_simbacore_simd_mode(M13_SIMD_RMS_BF16);
    start_simbacore_and_streamers(0, 0, 0, 0);
    wait_simbacore_and_streamer();

    // 2. rms = sqrt(rms)
    set_simd_streamer_no_b((uint32_t)ptr_rms, M12_R7_rms_ss, M12_R7_rms_tb, M12_R7_rms_ts,  //
                           (uint32_t)ptr_rms, M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);
    set_simbacore_simd_mode(M15_SIMD_SQRT_BF16);
    start_simbacore_and_streamers(0, 0, 0, 0);
    wait_simbacore_and_streamer();

    // 3. rms = sqrt(D) / rms = 1/rms_true   (reciprocal on the small per-token vector; sqrt(D)
    //    numerator folds in the "* 1/D" of the true rms)
    set_simd_streamer_csr((uint32_t)ptr_sqrtd, M12_R7_rms_ss, M12_R7_rms_tb, (int32_t*)zero_ts,  //
                          (uint32_t)ptr_rms, M12_R7_rms_ss, M12_R7_rms_tb, M12_R7_rms_ts,       //
                          (uint32_t)ptr_rms, M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);
    set_simbacore_simd_mode(M14_SIMD_DIV_BF16);
    start_simbacore_and_streamers(0, 0, 0, 0);
    wait_simbacore_and_streamer();

    // 4. x = x * rms   (1/rms broadcast over D, one per token)
    set_simd_streamer_csr((uint32_t)ptr_x, M12_R7_x_ss, M12_R7_x_tb, M12_R7_x_ts,                   //
                          (uint32_t)ptr_rms, M12_R13_x_rms_ss, M12_R13_x_rms_tb, M12_R13_x_rms_ts,  //
                          (uint32_t)ptr_x, M12_W3_x_ss, M12_W3_x_tb, M12_W3_x_ts);
    set_simbacore_simd_mode(M10_SIMD_MUL_BF16);
    start_simbacore_and_streamers(0, 0, 0, 0);
    wait_simbacore_and_streamer();

    // 5. x = x * weight'   (weight' = weight*sqrt(D), per channel, stationary over L)
    set_simd_streamer_csr((uint32_t)ptr_x, M12_R7_x_w_ss, M12_R7_x_w_tb, M12_R7_x_w_ts,          //
                          (uint32_t)ptr_weight, M12_R13_x_w_ss, M12_R13_x_w_tb, M12_R13_x_w_ts,  //
                          (uint32_t)ptr_x, M12_W3_x_w_ss, M12_W3_x_w_tb, M12_W3_x_w_ts);
    // (still in MUL mode)
    start_simbacore_and_streamers(0, 0, 0, 0);
    wait_simbacore_and_streamer();

    // Drain the SIMD write buffer to TCDM before the DM core DMAs this slot out.
    asm volatile("fence" ::: "memory");
}

int test() {
    int err = 0;

    // TCDM layout: resident weight + sqrt(D) constant + one rms scratch + two x slots.
    void* base           = snrt_l1_next();
    uint16_t* ptr_weight = (uint16_t*)base;
    uint16_t* ptr_sqrtd  = (uint16_t*)((uint8_t*)ptr_weight + M12_length_weight);
    uint16_t* ptr_rms    = (uint16_t*)((uint8_t*)ptr_sqrtd + simdLanes_bf16 * 2);
    uint16_t* ptr_x[2];
    ptr_x[0] = (uint16_t*)((uint8_t*)ptr_rms + M12_length_rms_tile);
    ptr_x[1] = (uint16_t*)((uint8_t*)ptr_x[0] + M12_length_x_tile);

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;

    // L3 staging buffer for the normalized output (full seqLen x dModel).
    static uint8_t* l3_out;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: RMSNorm tiled (seqLen=%u dModel=%u nb_tiles=%u L_tile=%u)\n\n", seqLen, dModel,
               nb_tiles, M12_L_tile);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();

        // Reserve 16 KiB so snrt_l3alloc does not overlap the runtime's putc_buffer.
        (void)snrt_l3alloc(16 * 1024);
        l3_out = (uint8_t*)snrt_l3alloc((uint32_t)nb_tiles * M12_length_x_tile);
    }

    // Load resident weight + first x tile.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d((uint8_t*)ptr_weight, (uint8_t*)M12_weight, M12_length_weight);
        snrt_dma_start_1d((uint8_t*)ptr_x[0], (uint8_t*)M12_x, M12_length_x_tile);
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
            if (n + 1 < nb_tiles)
                snrt_dma_start_1d((uint8_t*)ptr_x[other], (uint8_t*)M12_x + (n + 1) * M12_length_x_tile,
                                  M12_length_x_tile);
        }

        if (snrt_global_core_idx() == 0) {
            normalize_tile(ptr_x[s], ptr_rms, ptr_weight, ptr_sqrtd);
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

        printf("Test RMSNorm tiled: (%u x %u), nb_tiles=%u\n", seqLen, dModel, nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
