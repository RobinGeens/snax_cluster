// Copyright 2026 KU Leuven. memsim — per-hart architectural state.
#pragma once
#include <cstdint>
#include <unordered_map>

// CSR numbers we care about.
enum : uint32_t {
    CSR_FFLAGS = 0x001, CSR_FRM = 0x002, CSR_FCSR = 0x003,
    CSR_MSTATUS = 0x300, CSR_MISA = 0x301, CSR_MIE = 0x304, CSR_MTVEC = 0x305,
    CSR_MSCRATCH = 0x340, CSR_MEPC = 0x341, CSR_MCAUSE = 0x342,
    CSR_MTVAL = 0x343, CSR_MIP = 0x344,
    CSR_MCYCLE = 0xB00, CSR_MINSTRET = 0xB02, CSR_MCYCLEH = 0xB80,
    CSR_CYCLE = 0xC00, CSR_TIME = 0xC01,
    CSR_CLUSTER_BASE_L = 0xBC1, CSR_CLUSTER_BASE_H = 0xBC2, CSR_CORE_INFO = 0xBC3,
    CSR_HW_BARRIER = 0x7C2,
    CSR_MHARTID = 0xF14,
};

// SNAX CSR window (generated streamer_csr_addr_map.h + snax-simbacore-lib.h).
// Streamer block begins at 960; SimbaCore block is contiguous after it.
enum : uint32_t {
    SNAX_CSR_LO = 960,    // first streamer CSR
    SNAX_CSR_HI = 1175,   // ISCORE_TILE_CNT (last)
};

enum class HartState { Run, AtBarrier, BlockedPoll, Halted, Parked };

struct Hart {
    int id = 0;             // hart index (0 = compute, 1 = DM)
    uint32_t x[32] = {0};
    uint64_t f[32] = {0};   // FP regs (raw bits / integer reinterpretation)
    uint32_t pc = 0;
    uint64_t cycle = 0;     // this hart's simulated time
    HartState state = HartState::Run;
    std::unordered_map<uint32_t, uint32_t> csr;  // generic CSR backing store

    // Poll-loop detection: count consecutive instructions with no forward
    // progress past a backward branch while the world frontier is unchanged.
    uint32_t poll_spins = 0;
    uint32_t last_backjump_pc = 0xffffffff;

    // Load-use / mul-div scoreboard (Snitch is single-issue single-stage; the only
    // structural stall is a consumer reading a still-in-flight load/MUL/DIV result —
    // snitch.sv:455-475). NumOutstandingLoads=1 so a single pending dest suffices.
    uint32_t pend_reg = 0;      // GPR with an in-flight long-latency result (0 = none)
    uint64_t pend_ready = 0;    // cycle that result becomes consumable

    inline void set(uint32_t r, uint32_t v) { if (r) x[r] = v; }
    inline uint32_t get(uint32_t r) const { return x[r]; }
};

// Signal returned by one interpreter step, telling the scheduler what happened.
enum class Step {
    Normal,      // ordinary instruction retired
    Barrier,     // hit csrr x0,0x7C2 — hart now AtBarrier
    Halt,        // tohost exit written — whole sim should stop
    Wfi,         // wfi — park this hart
    PollRead,    // executed a RO read of world/htif state inside a wait loop
};
