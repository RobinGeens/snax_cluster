// Copyright 2026 KU Leuven. memsim — RV32IMA(+FP-as-int) functional interpreter.
#pragma once
#include "hart.hpp"
#include "machine.hpp"

// Per-hart staging registers for the multi-instruction xDMA sequence.
struct DmaStage {
    uint64_t src = 0, dst = 0;
    uint32_t dst_stride = 0, src_stride = 0, repeat = 1;
};

// Execute exactly one instruction for hart `h`. Returns a Step signal telling
// the scheduler about barriers / halts / polls. `dma` is the hart's xDMA
// staging state (persists across the dmsrc/dmdst/.../dmcpyi sequence).
Step interp_step(Machine& m, Hart& h, DmaStage& dma);
