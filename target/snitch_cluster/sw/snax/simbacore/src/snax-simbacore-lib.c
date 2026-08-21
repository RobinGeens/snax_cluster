// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "snax-simbacore-lib.h"
#include <stdint.h>
#include "streamer_csr_addr_map.h"

// Shorthand function to set only the streamers used in OSGeMM
void set_osgemm_streamer_csr(uint32_t A_ptr, int32_t* A_ss, int32_t* A_tb, int32_t* A_ts,  //
                             uint32_t B_ptr, int32_t* B_ss, int32_t* B_tb, int32_t* B_ts,  //
                             uint32_t D_ptr, int32_t* D_ss, int32_t* D_tb, int32_t* D_ts) {
    // osCore input R0
    _Static_assert(S_STRIDE_NUM_READER_0 == 1 && T_BOUND_NUM_READER_0 == 4 && T_STRIDE_NUM_READER_0 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_READER_0_LOW, A_ptr);         // Base ptr
    write_csr(S_STRIDE_BASE_READER_0 + 0, A_ss[0]);  // Spatial stride
    write_csr(T_BOUND_BASE_READER_0 + 0, A_tb[0]);   // Temporal bound
    write_csr(T_BOUND_BASE_READER_0 + 1, A_tb[1]);
    write_csr(T_BOUND_BASE_READER_0 + 2, A_tb[2]);
    write_csr(T_BOUND_BASE_READER_0 + 3, A_tb[3]);
    write_csr(T_STRIDE_BASE_READER_0 + 0, A_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_READER_0 + 1, A_ts[1]);
    write_csr(T_STRIDE_BASE_READER_0 + 2, A_ts[2]);
    write_csr(T_STRIDE_BASE_READER_0 + 3, A_ts[3]);

    // osCore weight R1
    _Static_assert(S_STRIDE_NUM_READER_1 == 1 && T_BOUND_NUM_READER_1 == 4 && T_STRIDE_NUM_READER_1 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_READER_1_LOW, B_ptr);         // Base ptr
    write_csr(S_STRIDE_BASE_READER_1 + 0, B_ss[0]);  // Spatial stride
    write_csr(T_BOUND_BASE_READER_1 + 0, B_tb[0]);   // Temporal bound
    write_csr(T_BOUND_BASE_READER_1 + 1, B_tb[1]);
    write_csr(T_BOUND_BASE_READER_1 + 2, B_tb[2]);
    write_csr(T_BOUND_BASE_READER_1 + 3, B_tb[3]);
    write_csr(T_STRIDE_BASE_READER_1 + 0, B_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_READER_1 + 1, B_ts[1]);
    write_csr(T_STRIDE_BASE_READER_1 + 2, B_ts[2]);
    write_csr(T_STRIDE_BASE_READER_1 + 3, B_ts[3]);

    // osCore output W0
    _Static_assert(S_STRIDE_NUM_WRITER_0 == 1 && T_BOUND_NUM_WRITER_0 == 4 && T_STRIDE_NUM_WRITER_0 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_WRITER_0_LOW, D_ptr);         // Base ptr
    write_csr(S_STRIDE_BASE_WRITER_0 + 0, D_ss[0]);  // Spatial stride
    write_csr(T_BOUND_BASE_WRITER_0 + 0, D_tb[0]);   // Temporal bound
    write_csr(T_BOUND_BASE_WRITER_0 + 1, D_tb[1]);
    write_csr(T_BOUND_BASE_WRITER_0 + 2, D_tb[2]);
    write_csr(T_BOUND_BASE_WRITER_0 + 3, D_tb[3]);
    write_csr(T_STRIDE_BASE_WRITER_0 + 0, D_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_WRITER_0 + 1, D_ts[1]);
    write_csr(T_STRIDE_BASE_WRITER_0 + 2, D_ts[2]);
    write_csr(T_STRIDE_BASE_WRITER_0 + 3, D_ts[3]);

    // Disable all other streamers by setting bound to 0
    write_csr(T_BOUND_BASE_READER_2, 0);
    write_csr(T_BOUND_BASE_READER_3, 0);
    write_csr(T_BOUND_BASE_READER_4, 0);
    write_csr(T_BOUND_BASE_READER_5, 0);
    write_csr(T_BOUND_BASE_READER_6, 0);
    write_csr(T_BOUND_BASE_READER_7, 0);
    write_csr(T_BOUND_BASE_READER_8, 0);
    write_csr(T_BOUND_BASE_READER_9, 0);
    write_csr(T_BOUND_BASE_READER_10, 0);
    write_csr(T_BOUND_BASE_READER_11, 0);
    write_csr(T_BOUND_BASE_READER_12, 0);
    write_csr(T_BOUND_BASE_READER_13, 0);
    write_csr(T_BOUND_BASE_WRITER_1, 0);
    write_csr(T_BOUND_BASE_WRITER_2, 0);
    write_csr(T_BOUND_BASE_WRITER_3, 0);
}

// Shorthand function to set only the streamers used in ISGEMM
void set_isgemm_streamer_csr(uint32_t A_ptr, int32_t* A_ss, int32_t* A_tb, int32_t* A_ts,  //
                             uint32_t B_ptr, int32_t* B_ss, int32_t* B_tb, int32_t* B_ts,  //
                             uint32_t CD_ptr, int32_t* CD_ss, int32_t* CD_tb, int32_t* CD_ts) {
    // iscore input R11
    _Static_assert(S_STRIDE_NUM_READER_11 == 1 && T_BOUND_NUM_READER_11 == 4 && T_STRIDE_NUM_READER_11 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_READER_11_LOW, A_ptr);         // Base ptr
    write_csr(S_STRIDE_BASE_READER_11 + 0, A_ss[0]);  // Spatial stride
    write_csr(T_BOUND_BASE_READER_11 + 0, A_tb[0]);   // Temporal bound
    write_csr(T_BOUND_BASE_READER_11 + 1, A_tb[1]);
    write_csr(T_BOUND_BASE_READER_11 + 2, A_tb[2]);
    write_csr(T_BOUND_BASE_READER_11 + 3, A_tb[3]);
    write_csr(T_STRIDE_BASE_READER_11 + 0, A_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_READER_11 + 1, A_ts[1]);
    write_csr(T_STRIDE_BASE_READER_11 + 2, A_ts[2]);
    write_csr(T_STRIDE_BASE_READER_11 + 3, A_ts[3]);

    // iscore weight R12
    _Static_assert(S_STRIDE_NUM_READER_12 == 1 && T_BOUND_NUM_READER_12 == 4 && T_STRIDE_NUM_READER_12 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_READER_12_LOW, B_ptr);         // Base ptr
    write_csr(S_STRIDE_BASE_READER_12 + 0, B_ss[0]);  // Spatial stride
    write_csr(T_BOUND_BASE_READER_12 + 0, B_tb[0]);   // Temporal bound
    write_csr(T_BOUND_BASE_READER_12 + 1, B_tb[1]);
    write_csr(T_BOUND_BASE_READER_12 + 2, B_tb[2]);
    write_csr(T_BOUND_BASE_READER_12 + 3, B_tb[3]);
    write_csr(T_STRIDE_BASE_READER_12 + 0, B_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_READER_12 + 1, B_ts[1]);
    write_csr(T_STRIDE_BASE_READER_12 + 2, B_ts[2]);
    write_csr(T_STRIDE_BASE_READER_12 + 3, B_ts[3]);

    // iscore bias/psum R13
    _Static_assert(S_STRIDE_NUM_READER_13 == 1 && T_BOUND_NUM_READER_13 == 4 && T_STRIDE_NUM_READER_13 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_READER_13_LOW, CD_ptr);         // Base ptr
    write_csr(S_STRIDE_BASE_READER_13 + 0, CD_ss[0]);  // Spatial stride
    write_csr(T_BOUND_BASE_READER_13 + 0, CD_tb[0]);   // Temporal bound
    write_csr(T_BOUND_BASE_READER_13 + 1, CD_tb[1]);
    write_csr(T_BOUND_BASE_READER_13 + 2, CD_tb[2]);
    write_csr(T_BOUND_BASE_READER_13 + 3, CD_tb[3]);
    write_csr(T_STRIDE_BASE_READER_13 + 0, CD_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_READER_13 + 1, CD_ts[1]);
    write_csr(T_STRIDE_BASE_READER_13 + 2, CD_ts[2]);
    write_csr(T_STRIDE_BASE_READER_13 + 3, CD_ts[3]);

    // iscore output W3
    _Static_assert(S_STRIDE_NUM_WRITER_3 == 2 && T_BOUND_NUM_WRITER_3 == 4 && T_STRIDE_NUM_WRITER_3 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_WRITER_3_LOW, CD_ptr);            // Base ptr
    write_csr(S_STRIDE_BASE_WRITER_3 + 0, CD_ss[0]);     // Spatial stride 0
    write_csr(S_STRIDE_BASE_WRITER_3 + 1, CD_ss[1]);     // Spatial stride 1
    write_csr(T_BOUND_BASE_WRITER_3 + 0, CD_tb[0]);  // Temporal bound
    write_csr(T_BOUND_BASE_WRITER_3 + 1, CD_tb[1]);
    write_csr(T_BOUND_BASE_WRITER_3 + 2, CD_tb[2]);
    write_csr(T_BOUND_BASE_WRITER_3 + 3, CD_tb[3]);
    write_csr(T_STRIDE_BASE_WRITER_3 + 0, CD_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_WRITER_3 + 1, CD_ts[1]);
    write_csr(T_STRIDE_BASE_WRITER_3 + 2, CD_ts[2]);
    write_csr(T_STRIDE_BASE_WRITER_3 + 3, CD_ts[3]);

    // Disable all other streamers by setting bound to 0
    write_csr(T_BOUND_BASE_READER_0, 0);
    write_csr(T_BOUND_BASE_READER_1, 0);
    write_csr(T_BOUND_BASE_READER_2, 0);
    write_csr(T_BOUND_BASE_READER_3, 0);
    write_csr(T_BOUND_BASE_READER_4, 0);
    write_csr(T_BOUND_BASE_READER_5, 0);
    write_csr(T_BOUND_BASE_READER_6, 0);
    write_csr(T_BOUND_BASE_READER_7, 0);
    write_csr(T_BOUND_BASE_READER_8, 0);
    write_csr(T_BOUND_BASE_READER_9, 0);
    write_csr(T_BOUND_BASE_READER_10, 0);
    write_csr(T_BOUND_BASE_WRITER_0, 0);
    write_csr(T_BOUND_BASE_WRITER_1, 0);
    write_csr(T_BOUND_BASE_WRITER_2, 0);
}

// Shorthand function to set only the streamers used in SIMD
void set_simd_streamer_csr(uint32_t A_ptr, int32_t* A_ss, int32_t* A_tb, int32_t* A_ts,  //
                           uint32_t B_ptr, int32_t* B_ss, int32_t* B_tb, int32_t* B_ts,  //
                           uint32_t C_ptr, int32_t* C_ss, int32_t* C_tb, int32_t* C_ts) {
    // Route input A to SUC BC: R7
    _Static_assert(S_STRIDE_NUM_READER_7 == 2 && T_BOUND_NUM_READER_7 == 4 && T_STRIDE_NUM_READER_7 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_READER_7_LOW, A_ptr);         // Base ptr
    write_csr(S_STRIDE_BASE_READER_7 + 0, A_ss[0]);  // Spatial stride 1
    write_csr(S_STRIDE_BASE_READER_7 + 1, A_ss[1]);  // Spatial stride 2
    write_csr(T_BOUND_BASE_READER_7 + 0, A_tb[0]);   // Temporal bound
    write_csr(T_BOUND_BASE_READER_7 + 1, A_tb[1]);
    write_csr(T_BOUND_BASE_READER_7 + 2, A_tb[2]);
    write_csr(T_BOUND_BASE_READER_7 + 3, A_tb[3]);
    write_csr(T_STRIDE_BASE_READER_7 + 0, A_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_READER_7 + 1, A_ts[1]);
    write_csr(T_STRIDE_BASE_READER_7 + 2, A_ts[2]);
    write_csr(T_STRIDE_BASE_READER_7 + 3, A_ts[3]);

    // Route input B to iscore psum: R13
    _Static_assert(S_STRIDE_NUM_READER_13 == 1 && T_BOUND_NUM_READER_13 == 4 && T_STRIDE_NUM_READER_13 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_READER_13_LOW, B_ptr);         // Base ptr
    write_csr(S_STRIDE_BASE_READER_13 + 0, B_ss[0]);  // Spatial stride
    write_csr(T_BOUND_BASE_READER_13 + 0, B_tb[0]);   // Temporal bound
    write_csr(T_BOUND_BASE_READER_13 + 1, B_tb[1]);
    write_csr(T_BOUND_BASE_READER_13 + 2, B_tb[2]);
    write_csr(T_BOUND_BASE_READER_13 + 3, B_tb[3]);
    write_csr(T_STRIDE_BASE_READER_13 + 0, B_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_READER_13 + 1, B_ts[1]);
    write_csr(T_STRIDE_BASE_READER_13 + 2, B_ts[2]);
    write_csr(T_STRIDE_BASE_READER_13 + 3, B_ts[3]);

    // Route output C to iscore out: W3
    _Static_assert(S_STRIDE_NUM_WRITER_3 == 2 && T_BOUND_NUM_WRITER_3 == 4 && T_STRIDE_NUM_WRITER_3 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_WRITER_3_LOW, C_ptr);         // Base ptr
    write_csr(S_STRIDE_BASE_WRITER_3 + 0, C_ss[0]);  // Spatial stride 0
    write_csr(S_STRIDE_BASE_WRITER_3 + 1, C_ss[1]);  // Spatial stride 1
    write_csr(T_BOUND_BASE_WRITER_3 + 0, C_tb[0]);   // Temporal bound
    write_csr(T_BOUND_BASE_WRITER_3 + 1, C_tb[1]);
    write_csr(T_BOUND_BASE_WRITER_3 + 2, C_tb[2]);
    write_csr(T_BOUND_BASE_WRITER_3 + 3, C_tb[3]);
    write_csr(T_STRIDE_BASE_WRITER_3 + 0, C_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_WRITER_3 + 1, C_ts[1]);
    write_csr(T_STRIDE_BASE_WRITER_3 + 2, C_ts[2]);
    write_csr(T_STRIDE_BASE_WRITER_3 + 3, C_ts[3]);

    // Disable all other streamers by setting bound to 0
    write_csr(T_BOUND_BASE_READER_0, 0);
    write_csr(T_BOUND_BASE_READER_1, 0);
    write_csr(T_BOUND_BASE_READER_2, 0);
    write_csr(T_BOUND_BASE_READER_3, 0);
    write_csr(T_BOUND_BASE_READER_4, 0);
    write_csr(T_BOUND_BASE_READER_5, 0);
    write_csr(T_BOUND_BASE_READER_6, 0);
    write_csr(T_BOUND_BASE_READER_8, 0);
    write_csr(T_BOUND_BASE_READER_9, 0);
    write_csr(T_BOUND_BASE_READER_10, 0);
    write_csr(T_BOUND_BASE_READER_11, 0);
    write_csr(T_BOUND_BASE_READER_12, 0);
    write_csr(T_BOUND_BASE_WRITER_0, 0);
    write_csr(T_BOUND_BASE_WRITER_1, 0);
    write_csr(T_BOUND_BASE_WRITER_2, 0);
}

// Shorthand function to set only the streamers used in SIMD
void set_simd_streamer_no_b(uint32_t A_ptr, int32_t* A_ss, int32_t* A_tb, int32_t* A_ts,  //
                            uint32_t C_ptr, int32_t* C_ss, int32_t* C_tb, int32_t* C_ts) {
    // Route input A to SUC BC: R7
    _Static_assert(S_STRIDE_NUM_READER_7 == 2 && T_BOUND_NUM_READER_7 == 4 && T_STRIDE_NUM_READER_7 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_READER_7_LOW, A_ptr);         // Base ptr
    write_csr(S_STRIDE_BASE_READER_7 + 0, A_ss[0]);  // Spatial stride
    write_csr(S_STRIDE_BASE_READER_7 + 1, A_ss[1]);  // Spatial stride
    write_csr(T_BOUND_BASE_READER_7 + 0, A_tb[0]);   // Temporal bound
    write_csr(T_BOUND_BASE_READER_7 + 1, A_tb[1]);
    write_csr(T_BOUND_BASE_READER_7 + 2, A_tb[2]);
    write_csr(T_BOUND_BASE_READER_7 + 3, A_tb[3]);
    write_csr(T_STRIDE_BASE_READER_7 + 0, A_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_READER_7 + 1, A_ts[1]);
    write_csr(T_STRIDE_BASE_READER_7 + 2, A_ts[2]);
    write_csr(T_STRIDE_BASE_READER_7 + 3, A_ts[3]);

    // Route output C to iscore out: W3
    _Static_assert(S_STRIDE_NUM_WRITER_3 == 2 && T_BOUND_NUM_WRITER_3 == 4 && T_STRIDE_NUM_WRITER_3 == 4,
                   "loop unroll mismatch");
    write_csr(BASE_PTR_WRITER_3_LOW, C_ptr);         // Base ptr
    write_csr(S_STRIDE_BASE_WRITER_3 + 0, C_ss[0]);  // Spatial stride 0
    write_csr(S_STRIDE_BASE_WRITER_3 + 1, C_ss[1]);  // Spatial stride 1
    write_csr(T_BOUND_BASE_WRITER_3 + 0, C_tb[0]);   // Temporal bound
    write_csr(T_BOUND_BASE_WRITER_3 + 1, C_tb[1]);
    write_csr(T_BOUND_BASE_WRITER_3 + 2, C_tb[2]);
    write_csr(T_BOUND_BASE_WRITER_3 + 3, C_tb[3]);
    write_csr(T_STRIDE_BASE_WRITER_3 + 0, C_ts[0]);  // Temporal stride
    write_csr(T_STRIDE_BASE_WRITER_3 + 1, C_ts[1]);
    write_csr(T_STRIDE_BASE_WRITER_3 + 2, C_ts[2]);
    write_csr(T_STRIDE_BASE_WRITER_3 + 3, C_ts[3]);

    // Disable all other streamers by setting bound to 0
    write_csr(T_BOUND_BASE_READER_0, 0);
    write_csr(T_BOUND_BASE_READER_1, 0);
    write_csr(T_BOUND_BASE_READER_2, 0);
    write_csr(T_BOUND_BASE_READER_3, 0);
    write_csr(T_BOUND_BASE_READER_4, 0);
    write_csr(T_BOUND_BASE_READER_5, 0);
    write_csr(T_BOUND_BASE_READER_6, 0);
    write_csr(T_BOUND_BASE_READER_8, 0);
    write_csr(T_BOUND_BASE_READER_9, 0);
    write_csr(T_BOUND_BASE_READER_10, 0);
    write_csr(T_BOUND_BASE_READER_11, 0);
    write_csr(T_BOUND_BASE_READER_12, 0);
    write_csr(T_BOUND_BASE_WRITER_0, 0);
    write_csr(T_BOUND_BASE_WRITER_1, 0);
    write_csr(T_BOUND_BASE_WRITER_2, 0);

    // Fully initialize R13 to a safe disabled state
    // The stride-zero bug forces bound=1 when stride=0, so all strides must be non-zero to keep R13 truly disabled.
    write_csr(S_STRIDE_BASE_READER_13, 1);
    write_csr(T_BOUND_BASE_READER_13 + 0, 0);
    write_csr(T_BOUND_BASE_READER_13 + 1, 0);
    write_csr(T_BOUND_BASE_READER_13 + 2, 0);
    write_csr(T_BOUND_BASE_READER_13 + 3, 0);
    write_csr(T_STRIDE_BASE_READER_13 + 0, 1);
    write_csr(T_STRIDE_BASE_READER_13 + 1, 1);
    write_csr(T_STRIDE_BASE_READER_13 + 2, 1);
    write_csr(T_STRIDE_BASE_READER_13 + 3, 1);
}

void set_streamer_csr(

    uint32_t R0_ptr, int32_t* R0_ss, int32_t* R0_tb, int32_t* R0_ts, bool R0_en,       // osCore in
    uint32_t R1_ptr, int32_t* R1_ss, int32_t* R1_tb, int32_t* R1_ts, bool R1_en,       // oscore weight
    uint32_t R2_ptr, int32_t* R2_ss, int32_t* R2_tb, int32_t* R2_ts, bool R2_en,       // switchCore/ in
    uint32_t R3_ptr, int32_t* R3_ss, int32_t* R3_tb, int32_t* R3_ts, bool R3_en,       // switchCore weight
    uint32_t R4_ptr, int32_t* R4_ss, int32_t* R4_tb, int32_t* R4_ts, bool R4_en,       // switchCore bias
    uint32_t R5_ptr, int32_t* R5_ss, int32_t* R5_tb, int32_t* R5_ts, bool R5_en,       // switchCore  matmul weight
    uint32_t R6_ptr, int32_t* R6_ss, int32_t* R6_tb, int32_t* R6_ts, bool R6_en,       // SUC A
    uint32_t R7_ptr, int32_t* R7_ss, int32_t* R7_tb, int32_t* R7_ts, bool R7_en,       // SUC BC
    uint32_t R8_ptr, int32_t* R8_ss, int32_t* R8_tb, int32_t* R8_ts, bool R8_en,       // SUC D
    uint32_t R9_ptr, int32_t* R9_ss, int32_t* R9_tb, int32_t* R9_ts, bool R9_en,       // SUC x
    uint32_t R10_ptr, int32_t* R10_ss, int32_t* R10_tb, int32_t* R10_ts, bool R10_en,  // SUC z
    uint32_t R11_ptr, int32_t* R11_ss, int32_t* R11_tb, int32_t* R11_ts, bool R11_en,  // iscore in
    uint32_t R12_ptr, int32_t* R12_ss, int32_t* R12_tb, int32_t* R12_ts, bool R12_en,  // isCore weight
    uint32_t R13_ptr, int32_t* R13_ss, int32_t* R13_tb, int32_t* R13_ts, bool R13_en,  // isCore psum

    uint32_t W0_ptr, int32_t* W0_ss, int32_t* W0_tb, int32_t* W0_ts, bool W0_en,  // osCore out
    uint32_t W1_ptr, int32_t* W1_ss, int32_t* W1_tb, int32_t* W1_ts, bool W1_en,  // switchCore out
    uint32_t W2_ptr, int32_t* W2_ss, int32_t* W2_tb, int32_t* W2_ts, bool W2_en,  // SUC out
    uint32_t W3_ptr, int32_t* W3_ss, int32_t* W3_tb, int32_t* W3_ts, bool W3_en   // isCore out
) {
    if (R0_en) {
        _Static_assert(S_STRIDE_NUM_READER_0 == 1 && T_BOUND_NUM_READER_0 == 4 && T_STRIDE_NUM_READER_0 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_0_LOW, R0_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_0 + 0, R0_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_0 + 0, R0_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_0 + 1, R0_tb[1]);
        write_csr(T_BOUND_BASE_READER_0 + 2, R0_tb[2]);
        write_csr(T_BOUND_BASE_READER_0 + 3, R0_tb[3]);
        write_csr(T_STRIDE_BASE_READER_0 + 0, R0_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_0 + 1, R0_ts[1]);
        write_csr(T_STRIDE_BASE_READER_0 + 2, R0_ts[2]);
        write_csr(T_STRIDE_BASE_READER_0 + 3, R0_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_0, 0);
    }

    if (R1_en) {
        _Static_assert(S_STRIDE_NUM_READER_1 == 1 && T_BOUND_NUM_READER_1 == 4 && T_STRIDE_NUM_READER_1 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_1_LOW, R1_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_1 + 0, R1_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_1 + 0, R1_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_1 + 1, R1_tb[1]);
        write_csr(T_BOUND_BASE_READER_1 + 2, R1_tb[2]);
        write_csr(T_BOUND_BASE_READER_1 + 3, R1_tb[3]);
        write_csr(T_STRIDE_BASE_READER_1 + 0, R1_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_1 + 1, R1_ts[1]);
        write_csr(T_STRIDE_BASE_READER_1 + 2, R1_ts[2]);
        write_csr(T_STRIDE_BASE_READER_1 + 3, R1_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_1, 0);
    }

    if (R2_en) {
        _Static_assert(S_STRIDE_NUM_READER_2 == 1 && T_BOUND_NUM_READER_2 == 4 && T_STRIDE_NUM_READER_2 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_2_LOW, R2_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_2 + 0, R2_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_2 + 0, R2_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_2 + 1, R2_tb[1]);
        write_csr(T_BOUND_BASE_READER_2 + 2, R2_tb[2]);
        write_csr(T_BOUND_BASE_READER_2 + 3, R2_tb[3]);
        write_csr(T_STRIDE_BASE_READER_2 + 0, R2_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_2 + 1, R2_ts[1]);
        write_csr(T_STRIDE_BASE_READER_2 + 2, R2_ts[2]);
        write_csr(T_STRIDE_BASE_READER_2 + 3, R2_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_2, 0);
    }

    if (R3_en) {
        _Static_assert(S_STRIDE_NUM_READER_3 == 1 && T_BOUND_NUM_READER_3 == 4 && T_STRIDE_NUM_READER_3 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_3_LOW, R3_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_3 + 0, R3_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_3 + 0, R3_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_3 + 1, R3_tb[1]);
        write_csr(T_BOUND_BASE_READER_3 + 2, R3_tb[2]);
        write_csr(T_BOUND_BASE_READER_3 + 3, R3_tb[3]);
        write_csr(T_STRIDE_BASE_READER_3 + 0, R3_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_3 + 1, R3_ts[1]);
        write_csr(T_STRIDE_BASE_READER_3 + 2, R3_ts[2]);
        write_csr(T_STRIDE_BASE_READER_3 + 3, R3_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_3, 0);
    }

    if (R4_en) {
        _Static_assert(S_STRIDE_NUM_READER_4 == 1 && T_BOUND_NUM_READER_4 == 4 && T_STRIDE_NUM_READER_4 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_4_LOW, R4_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_4 + 0, R4_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_4 + 0, R4_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_4 + 1, R4_tb[1]);
        write_csr(T_BOUND_BASE_READER_4 + 2, R4_tb[2]);
        write_csr(T_BOUND_BASE_READER_4 + 3, R4_tb[3]);
        write_csr(T_STRIDE_BASE_READER_4 + 0, R4_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_4 + 1, R4_ts[1]);
        write_csr(T_STRIDE_BASE_READER_4 + 2, R4_ts[2]);
        write_csr(T_STRIDE_BASE_READER_4 + 3, R4_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_4, 0);
    }

    if (R5_en) {
        _Static_assert(S_STRIDE_NUM_READER_5 == 1 && T_BOUND_NUM_READER_5 == 4 && T_STRIDE_NUM_READER_5 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_5_LOW, R5_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_5 + 0, R5_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_5 + 0, R5_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_5 + 1, R5_tb[1]);
        write_csr(T_BOUND_BASE_READER_5 + 2, R5_tb[2]);
        write_csr(T_BOUND_BASE_READER_5 + 3, R5_tb[3]);
        write_csr(T_STRIDE_BASE_READER_5 + 0, R5_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_5 + 1, R5_ts[1]);
        write_csr(T_STRIDE_BASE_READER_5 + 2, R5_ts[2]);
        write_csr(T_STRIDE_BASE_READER_5 + 3, R5_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_5, 0);
    }

    if (R6_en) {
        _Static_assert(S_STRIDE_NUM_READER_6 == 1 && T_BOUND_NUM_READER_6 == 4 && T_STRIDE_NUM_READER_6 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_6_LOW, R6_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_6 + 0, R6_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_6 + 0, R6_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_6 + 1, R6_tb[1]);
        write_csr(T_BOUND_BASE_READER_6 + 2, R6_tb[2]);
        write_csr(T_BOUND_BASE_READER_6 + 3, R6_tb[3]);
        write_csr(T_STRIDE_BASE_READER_6 + 0, R6_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_6 + 1, R6_ts[1]);
        write_csr(T_STRIDE_BASE_READER_6 + 2, R6_ts[2]);
        write_csr(T_STRIDE_BASE_READER_6 + 3, R6_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_6, 0);
    }

    if (R7_en) {
        _Static_assert(S_STRIDE_NUM_READER_7 == 2 && T_BOUND_NUM_READER_7 == 4 && T_STRIDE_NUM_READER_7 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_7_LOW, R7_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_7 + 0, R7_ss[0]);  // Spatial stride 0
        write_csr(S_STRIDE_BASE_READER_7 + 1, R7_ss[1]);  // Spatial stride 1
        write_csr(T_BOUND_BASE_READER_7 + 0, R7_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_7 + 1, R7_tb[1]);
        write_csr(T_BOUND_BASE_READER_7 + 2, R7_tb[2]);
        write_csr(T_BOUND_BASE_READER_7 + 3, R7_tb[3]);
        write_csr(T_STRIDE_BASE_READER_7 + 0, R7_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_7 + 1, R7_ts[1]);
        write_csr(T_STRIDE_BASE_READER_7 + 2, R7_ts[2]);
        write_csr(T_STRIDE_BASE_READER_7 + 3, R7_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_7, 0);
    }

    if (R8_en) {
        _Static_assert(S_STRIDE_NUM_READER_8 == 1 && T_BOUND_NUM_READER_8 == 4 && T_STRIDE_NUM_READER_8 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_8_LOW, R8_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_8 + 0, R8_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_8 + 0, R8_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_8 + 1, R8_tb[1]);
        write_csr(T_BOUND_BASE_READER_8 + 2, R8_tb[2]);
        write_csr(T_BOUND_BASE_READER_8 + 3, R8_tb[3]);
        write_csr(T_STRIDE_BASE_READER_8 + 0, R8_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_8 + 1, R8_ts[1]);
        write_csr(T_STRIDE_BASE_READER_8 + 2, R8_ts[2]);
        write_csr(T_STRIDE_BASE_READER_8 + 3, R8_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_8, 0);
    }

    if (R9_en) {
        _Static_assert(S_STRIDE_NUM_READER_9 == 1 && T_BOUND_NUM_READER_9 == 4 && T_STRIDE_NUM_READER_9 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_9_LOW, R9_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_9 + 0, R9_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_9 + 0, R9_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_9 + 1, R9_tb[1]);
        write_csr(T_BOUND_BASE_READER_9 + 2, R9_tb[2]);
        write_csr(T_BOUND_BASE_READER_9 + 3, R9_tb[3]);
        write_csr(T_STRIDE_BASE_READER_9 + 0, R9_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_9 + 1, R9_ts[1]);
        write_csr(T_STRIDE_BASE_READER_9 + 2, R9_ts[2]);
        write_csr(T_STRIDE_BASE_READER_9 + 3, R9_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_9, 0);
    }

    if (R10_en) {
        _Static_assert(S_STRIDE_NUM_READER_10 == 1 && T_BOUND_NUM_READER_10 == 4 && T_STRIDE_NUM_READER_10 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_10_LOW, R10_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_10 + 0, R10_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_10 + 0, R10_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_10 + 1, R10_tb[1]);
        write_csr(T_BOUND_BASE_READER_10 + 2, R10_tb[2]);
        write_csr(T_BOUND_BASE_READER_10 + 3, R10_tb[3]);
        write_csr(T_STRIDE_BASE_READER_10 + 0, R10_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_10 + 1, R10_ts[1]);
        write_csr(T_STRIDE_BASE_READER_10 + 2, R10_ts[2]);
        write_csr(T_STRIDE_BASE_READER_10 + 3, R10_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_10, 0);
    }

    if (R11_en) {
        _Static_assert(S_STRIDE_NUM_READER_11 == 1 && T_BOUND_NUM_READER_11 == 4 && T_STRIDE_NUM_READER_11 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_11_LOW, R11_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_11 + 0, R11_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_11 + 0, R11_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_11 + 1, R11_tb[1]);
        write_csr(T_BOUND_BASE_READER_11 + 2, R11_tb[2]);
        write_csr(T_BOUND_BASE_READER_11 + 3, R11_tb[3]);
        write_csr(T_STRIDE_BASE_READER_11 + 0, R11_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_11 + 1, R11_ts[1]);
        write_csr(T_STRIDE_BASE_READER_11 + 2, R11_ts[2]);
        write_csr(T_STRIDE_BASE_READER_11 + 3, R11_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_11, 0);
    }

    if (R12_en) {
        _Static_assert(S_STRIDE_NUM_READER_12 == 1 && T_BOUND_NUM_READER_12 == 4 && T_STRIDE_NUM_READER_12 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_12_LOW, R12_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_12 + 0, R12_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_12 + 0, R12_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_12 + 1, R12_tb[1]);
        write_csr(T_BOUND_BASE_READER_12 + 2, R12_tb[2]);
        write_csr(T_BOUND_BASE_READER_12 + 3, R12_tb[3]);
        write_csr(T_STRIDE_BASE_READER_12 + 0, R12_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_12 + 1, R12_ts[1]);
        write_csr(T_STRIDE_BASE_READER_12 + 2, R12_ts[2]);
        write_csr(T_STRIDE_BASE_READER_12 + 3, R12_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_12, 0);
    }

    if (R13_en) {
        _Static_assert(S_STRIDE_NUM_READER_13 == 1 && T_BOUND_NUM_READER_13 == 4 && T_STRIDE_NUM_READER_13 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_READER_13_LOW, R13_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_READER_13 + 0, R13_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_READER_13 + 0, R13_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_READER_13 + 1, R13_tb[1]);
        write_csr(T_BOUND_BASE_READER_13 + 2, R13_tb[2]);
        write_csr(T_BOUND_BASE_READER_13 + 3, R13_tb[3]);
        write_csr(T_STRIDE_BASE_READER_13 + 0, R13_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_READER_13 + 1, R13_ts[1]);
        write_csr(T_STRIDE_BASE_READER_13 + 2, R13_ts[2]);
        write_csr(T_STRIDE_BASE_READER_13 + 3, R13_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_READER_13, 0);
    }

    if (W0_en) {
        _Static_assert(S_STRIDE_NUM_WRITER_0 == 1 && T_BOUND_NUM_WRITER_0 == 4 && T_STRIDE_NUM_WRITER_0 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_WRITER_0_LOW, W0_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_WRITER_0 + 0, W0_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_WRITER_0 + 0, W0_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_WRITER_0 + 1, W0_tb[1]);
        write_csr(T_BOUND_BASE_WRITER_0 + 2, W0_tb[2]);
        write_csr(T_BOUND_BASE_WRITER_0 + 3, W0_tb[3]);
        write_csr(T_STRIDE_BASE_WRITER_0 + 0, W0_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_WRITER_0 + 1, W0_ts[1]);
        write_csr(T_STRIDE_BASE_WRITER_0 + 2, W0_ts[2]);
        write_csr(T_STRIDE_BASE_WRITER_0 + 3, W0_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_WRITER_0, 0);
    }

    if (W1_en) {
        _Static_assert(S_STRIDE_NUM_WRITER_1 == 1 && T_BOUND_NUM_WRITER_1 == 4 && T_STRIDE_NUM_WRITER_1 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_WRITER_1_LOW, W1_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_WRITER_1 + 0, W1_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_WRITER_1 + 0, W1_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_WRITER_1 + 1, W1_tb[1]);
        write_csr(T_BOUND_BASE_WRITER_1 + 2, W1_tb[2]);
        write_csr(T_BOUND_BASE_WRITER_1 + 3, W1_tb[3]);
        write_csr(T_STRIDE_BASE_WRITER_1 + 0, W1_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_WRITER_1 + 1, W1_ts[1]);
        write_csr(T_STRIDE_BASE_WRITER_1 + 2, W1_ts[2]);
        write_csr(T_STRIDE_BASE_WRITER_1 + 3, W1_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_WRITER_1, 0);
    }

    if (W2_en) {
        _Static_assert(S_STRIDE_NUM_WRITER_2 == 1 && T_BOUND_NUM_WRITER_2 == 4 && T_STRIDE_NUM_WRITER_2 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_WRITER_2_LOW, W2_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_WRITER_2 + 0, W2_ss[0]);  // Spatial stride
        write_csr(T_BOUND_BASE_WRITER_2 + 0, W2_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_WRITER_2 + 1, W2_tb[1]);
        write_csr(T_BOUND_BASE_WRITER_2 + 2, W2_tb[2]);
        write_csr(T_BOUND_BASE_WRITER_2 + 3, W2_tb[3]);
        write_csr(T_STRIDE_BASE_WRITER_2 + 0, W2_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_WRITER_2 + 1, W2_ts[1]);
        write_csr(T_STRIDE_BASE_WRITER_2 + 2, W2_ts[2]);
        write_csr(T_STRIDE_BASE_WRITER_2 + 3, W2_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_WRITER_2, 0);
    }

    if (W3_en) {
        _Static_assert(S_STRIDE_NUM_WRITER_3 == 2 && T_BOUND_NUM_WRITER_3 == 4 && T_STRIDE_NUM_WRITER_3 == 4,
                       "loop unroll mismatch");
        write_csr(BASE_PTR_WRITER_3_LOW, W3_ptr);         // Base ptr
        write_csr(S_STRIDE_BASE_WRITER_3 + 0, W3_ss[0]);  // Spatial stride 0
        write_csr(S_STRIDE_BASE_WRITER_3 + 1, W3_ss[1]);  // Spatial stride 1
        write_csr(T_BOUND_BASE_WRITER_3 + 0, W3_tb[0]);   // Temporal bound
        write_csr(T_BOUND_BASE_WRITER_3 + 1, W3_tb[1]);
        write_csr(T_BOUND_BASE_WRITER_3 + 2, W3_tb[2]);
        write_csr(T_BOUND_BASE_WRITER_3 + 3, W3_tb[3]);
        write_csr(T_STRIDE_BASE_WRITER_3 + 0, W3_ts[0]);  // Temporal stride
        write_csr(T_STRIDE_BASE_WRITER_3 + 1, W3_ts[1]);
        write_csr(T_STRIDE_BASE_WRITER_3 + 2, W3_ts[2]);
        write_csr(T_STRIDE_BASE_WRITER_3 + 3, W3_ts[3]);
    } else {
        write_csr(T_BOUND_BASE_WRITER_3, 0);
    }
}

void set_simbacore_csr(uint32_t mode, uint32_t seqLen, uint32_t dModel, uint32_t dInner, uint32_t dtRank,
                       uint32_t dFinal) {
    write_csr(MODE, mode);
    write_csr(SEQ_LEN, seqLen);
    write_csr(D_MODEL, dModel);
    write_csr(D_INNER, dInner);
    write_csr(DT_RANK, dtRank);
    write_csr(D_FINAL, dFinal);
}

// Start the Streamer, including the two delayed Streamers (R10 and R11).
void start_simbacore_and_streamers(bool R10_en, uint32_t R10_start_cnt, bool R11_en, uint32_t R11_start_cnt) {
    _set_streamer_start();
    _set_simbacore_start();  // SimbaCore must start before the delayed streamers can start

    if (R10_en) {
        while (read_csr(R10_DELAY_GAUGE) < R10_start_cnt);
        write_csr(DELAYED_START_READER_10, 1);
#ifdef VERBOSE
        printf("[%d cc] Streamer R10 can start\r\n", snrt_mcycle());
#endif
    }

    if (R11_en) {
        while (read_csr(R11_DELAY_GAUGE) < R11_start_cnt);
        write_csr(DELAYED_START_READER_11, 1);
#ifdef VERBOSE
        printf("[%d cc] Streamer R11 can start\r\n", snrt_mcycle());
#endif
    }
}

// Stall until Streamer and GEMM accelerator finish
void wait_simbacore_and_streamer() {
#ifdef VERBOSE
    printf("[%d cc] Waiting for SimbaCore to finish...\r\n", snrt_mcycle());
#endif
    write_csr(STREAMER_START_CSR, 0);
    write_csr(SIMBACORE_START, 0);
    write_csr(DELAYED_START_READER_10, 0);
    write_csr(DELAYED_START_READER_11, 0);
    while (read_csr(SIMBACORE_BUSY));  // 1185 = 0x4a1
#ifdef VERBOSE
    printf("[%d cc] SimbaCore has finished. Waiting for Streamers...\r\n", snrt_mcycle());
#endif
    while (read_csr(STREAMER_BUSY_CSR));  // 1177 = 0x499
#ifdef VERBOSE
    printf("[%d cc] Streamers and SimbaCore have finished\r\n", snrt_mcycle());
#endif
}

// Read performance counter of the Streamer, a read-only CSR
uint32_t read_streamer_perf_counter() {
    uint32_t perf_counter = read_csr(STREAMER_PERFORMANCE_COUNTER_CSR);
    return perf_counter;
}

// Read performance counter of GEMM, a read-only CSR
uint32_t read_simbacore_perf_counter() {
    uint32_t perf_counter = read_csr(SIMBACORE_PERFORMANCE_COUNTER);
    return perf_counter;
}

static const int32_t TOLERANCE = 1;

static inline bool values_match_u8_lsb(uint8_t output_value, uint8_t golden_value) {
    if ((output_value == 0 && golden_value == 128) ||  // +0 == -0
        (output_value == 128 && golden_value == 0))
        return true;

    int16_t diff = (int16_t)output_value - (int16_t)golden_value;
    return (diff >= -TOLERANCE) && (diff <= TOLERANCE);
}

static inline bool values_match_u16_lsb(uint16_t output_value, uint16_t golden_value) {
    if ((output_value == 0 && golden_value == 0x8000u) ||  // +0 == -0
        (output_value == 0x8000u && golden_value == 0))
        return true;

    int32_t diff = (int32_t)output_value - (int32_t)golden_value;
    return (diff >= -TOLERANCE) && (diff <= TOLERANCE);
}

// Check result, word-by-word. data_length in bytes
uint32_t check_result_all(uint8_t* output, uint8_t* output_golden, int32_t data_length) {
    uint32_t err         = 0;
    int32_t num_elements = data_length / sizeof(uint8_t);
    printf("Checking results: %d bytes (%d elements)\r\n", data_length, num_elements);

    for (int i = 0; i < num_elements; i++) {
        uint8_t output_value = output[i];
        uint8_t golden_value = output_golden[i];
        if (!values_match_u8_lsb(output_value, golden_value)) {
            err++;
            printf("FAIL out[%d] = %d,\tref = %d\r\n", i, output_value, golden_value);
        } else {
            printf("PASS out[%d] = %d,\tref = %d\r\n", i, output_value, golden_value);
        }
    }
    return err;
}

uint32_t check_result_all_u16(uint16_t* output, uint16_t* output_golden, int32_t data_length) {
    uint32_t err         = 0;
    int32_t num_elements = data_length / sizeof(uint16_t);
    printf("Checking results (u16): %d bytes (%d elements)\r\n", data_length, num_elements);

    for (int i = 0; i < num_elements; i++) {
        uint16_t output_value = output[i];
        uint16_t golden_value = output_golden[i];
        if (!values_match_u16_lsb(output_value, golden_value)) {
            err++;
            printf("FAIL out[%d] = %u,\tref = %u\r\n", i, output_value, golden_value);
        } else {
            printf("PASS out[%d] = %u,\tref = %u\r\n", i, output_value, golden_value);
        }
    }
    return err;
}

// Check some samples of ther result to speed up verification
uint32_t check_result_sample_verbose(uint8_t* output, uint8_t* output_golden, int32_t* sample_indices,
                                     int32_t test_sample_count, const char* tensor_name, bool verbose) {
    uint32_t err = 0;
    printf("Checking results: sampling %d elements\r\n", test_sample_count);

    for (int i = 0; i < test_sample_count; i++) {
        int sample_index     = sample_indices[i];
        uint8_t output_value = output[sample_index];
        uint8_t golden_value = output_golden[sample_index];
        if (values_match_u8_lsb(output_value, golden_value)) {
            if (verbose)
                printf("PASS %s[%d] = %d,\tref = %d\r\n", tensor_name, sample_index, output_value, golden_value);
        } else {
            err++;
            if (verbose)
                printf("FAIL %s[%d] = %d,\tref = %d\r\n", tensor_name, sample_index, output_value, golden_value);
        }
    }
    return err;
}

uint32_t check_result_sample(uint8_t* output, uint8_t* output_golden, int32_t* sample_indices,
                             int32_t test_sample_count, const char* tensor_name) {
    return check_result_sample_verbose(output, output_golden, sample_indices, test_sample_count, tensor_name, true);
}

// Check some samples interpreting the buffers as uint16_t elements
uint32_t check_result_sample_u16_verbose(uint16_t* output, uint16_t* output_golden, int32_t* sample_indices,
                                         int32_t test_sample_count, const char* tensor_name, bool verbose) {
    uint32_t err = 0;
    printf("Checking results (u16): sampling %d elements\r\n", test_sample_count);

    for (int i = 0; i < test_sample_count; i++) {
        int sample_index      = sample_indices[i];
        uint16_t output_value = output[sample_index];
        uint16_t golden_value = output_golden[sample_index];
        if (!values_match_u16_lsb(output_value, golden_value)) {
            err++;
            if (verbose)
                printf("FAIL %s[%d] = %u,\tref = %u\r\n", tensor_name, sample_index, output_value, golden_value);
        } else {
            if (verbose)
                printf("PASS %s[%d] = %u,\tref = %u\r\n", tensor_name, sample_index, output_value, golden_value);
        }
    }
    return err;
}

uint32_t check_result_sample_u16(uint16_t* output, uint16_t* output_golden, int32_t* sample_indices,
                                 int32_t test_sample_count, const char* tensor_name) {
    return check_result_sample_u16_verbose(output, output_golden, sample_indices, test_sample_count, tensor_name, true);
}

uint32_t check_result_sample_swz(const uint8_t* output, const uint8_t* output_golden, const int32_t* sample_indices,
                                 const int32_t* sample_indices_swz, int32_t test_sample_count,
                                 const char* tensor_name) {
    uint32_t err = 0;
    printf("Checking results: sampling %d elements\r\n", test_sample_count);
    for (int i = 0; i < test_sample_count; i++) {
        uint8_t output_value = output[sample_indices_swz[i]];
        uint8_t golden_value = output_golden[sample_indices[i]];
        if (!values_match_u8_lsb(output_value, golden_value)) {
            err++;
            printf("FAIL %s[%d] = %d,\tref = %d\r\n", tensor_name, sample_indices[i], output_value, golden_value);
        }
    }
    return err;
}

// Initialize cycle counter (call once at program start)
void init_cycle_counter(void) {
    // Reset and start the cycle counter
    snrt_reset_perf_counter(SNRT_PERF_CNT0);
    snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_CYCLES, 0);
}
