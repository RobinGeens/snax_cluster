// Copyright 2026 KU Leuven. memsim — SNAX streamer/SimbaCore CSR numbers.
// Mirrors target/snitch_cluster/sw/snax/simbacore/include/streamer_csr_addr_map.h
// and snax-simbacore-lib.h (keep in sync if those are regenerated).
#pragma once
#include <cstdint>

enum : uint32_t {
    // Streamer per-port config block: readers R0..R13 then writers W0..W3,
    // each 11 CSRs (base_ptr_lo/hi, s_stride(s), t_bound[4], t_stride[4]),
    // except R7 which has 2 spatial strides (12 CSRs). Range [960, 1158].
    SNAX_STREAMER_CFG_LO = 960,
    SNAX_STREAMER_CFG_HI = 1158,

    SNAX_DELAYED_START_R10 = 1159,
    SNAX_DELAYED_START_R11 = 1160,
    SNAX_STREAMER_START = 1161,
    SNAX_STREAMER_BUSY = 1162,   // RO
    SNAX_STREAMER_PERF = 1163,   // RO

    // SimbaCore block = STREAMER_PERFORMANCE_COUNTER_CSR + 1 = 1164.
    SNAX_MODE = 1164,
    SNAX_SEQ_LEN = 1165,
    SNAX_D_MODEL = 1166,
    SNAX_DT_RANK = 1167,
    SNAX_D_INNER = 1168,
    SNAX_D_FINAL = 1169,
    SNAX_SIMBACORE_START = 1170,
    SNAX_SIMBACORE_BUSY = 1171,   // RO
    SNAX_SIMBACORE_PERF = 1172,   // RO
    SNAX_R10_GAUGE = 1173,        // RO: osCore output tiles
    SNAX_R11_GAUGE = 1174,        // RO: SUC output elements
    SNAX_ISCORE_TILE_CNT = 1175,  // RO: isCore output tiles
};

// Offload latency for a SNAX CSR *write* (config/base-ptr/start), in core cycles.
// Derived from the offload path RTL: a `csrw` with rd=x0 has no
// scoreboard wait; the offload spill register accepts when !a_full (ready same
// cycle, spill_register_flushable.sv) and ReqRspManager's config-write req_ready is
// combinational (ReqRspManager.sv:161-166). So back-to-back SNAX CSR writes retire
// at 1/cc (confirmed by vsim back-to-back issue = 1cc).
static constexpr unsigned SNAX_CSR_WRITE_COST = 1;

// Snitch in-order single-issue scalar stalls, derived from the core+cluster RTL for this
// build (cfg register_core_req/rsp=true, register_offload_req/rsp=true, RegisterTCDMCuts=0,
// TCDM Latency=1). These are charged to the CONSUMER of an in-flight result (interp.cpp
// scoreboard), so independent instructions stay 1 cc. They lift the Snitch wall-clock TOTAL
// (scalar glue between accelerator invocations); they never touch the accelerator perf
// counter (accel_end_), so SimbaCore is unchanged.
//   load-use: register_core_req(1) + TCDM read Latency(1) + register_core_rsp(1)
static constexpr unsigned LOAD_USE_STALL = 3;
//   MUL: register_offload_req(1) + 1 multiplier stage + register_offload_rsp(1)
static constexpr unsigned MUL_STALL = 3;
//   DIV/REM: serial divider envelope 2 + div_shift(<=32) + 2; full-width worst case (the
//   exact div_shift is operand-dependent -> residual; the 32-cc envelope is the derived bound)
static constexpr unsigned DIV_STALL = 32;
// HW cluster barrier rendezvous+release beyond the barrier CSR's own 1 cc: arrival latch (1)
// + csr_stall release (1) (snitch_barrier.sv:24,35 + snitch.sv:328). Independent of NrCores.
static constexpr unsigned HW_BARRIER_RELEASE_COST = 2;
