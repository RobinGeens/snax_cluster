// Copyright 2026 KU Leuven. memsim — machine context: memory, harts, htif,
// and the pluggable timing World interface.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "hart.hpp"
#include "mem.hpp"

// ---- HW-facing op descriptors shared between interpreter and World ----------
struct DmaDesc {
    uint64_t src = 0, dst = 0;
    uint32_t size = 0;                 // bytes (innermost extent for 2D)
    uint32_t dst_stride = 0, src_stride = 0, repeat = 1;
    bool is_2d = false;
    bool src_is_l3 = false, dst_is_l3 = false;
    int hart = 1;
};

struct Agu {                            // one streamer port descriptor
    uint64_t base = 0;
    int32_t s_stride[2] = {0, 0};
    int n_s = 0;
    int32_t t_bound[4] = {0, 0, 0, 0};
    int32_t t_stride[4] = {0, 0, 0, 0};
    int remap = 0;                      // ADDR_REMAP_INDEX: 0=identity, 1=XOR bank swizzle, 2=half
    bool enabled = false;
};

// AGU address remap (AddressGenUnit.scala): applied to every generated address.
//   1: addr[7:5] ^= addr[10:8]   2: addr[6:5] ^= addr[9:8]
inline uint32_t agu_swz(uint32_t a, int remap) {
    if (remap == 1) return a ^ ((a >> 3) & 0xE0u);
    if (remap == 2) return a ^ ((a >> 3) & 0x60u);
    return a;
}

struct SimbacoreCfg {
    uint32_t mode = 0, seqLen = 0, dModel = 0, dtRank = 0, dInner = 0, dFinal = 0;
};

// ---- World: the timing + functional datapath model -------------------------
// Base class is a no-op; SimWorld in world.cpp overrides it. All SNAX streamer/
// SimbaCore CSR accesses are funnelled through snax_read/write so the streamer-
// config + start + gauge logic lives in one place.
struct World {
    Memory* mem = nullptr;
    virtual ~World() = default;
    // time
    virtual void advance_to(uint64_t /*cycle*/) {}
    virtual uint64_t next_event_cycle() const { return UINT64_MAX; }
    // DMA (xDMA custom opcodes, not CSRs)
    virtual uint32_t dma_submit(const DmaDesc&, uint64_t /*at*/) { return 0; }
    virtual uint32_t dma_busy(uint64_t /*at*/) { return 0; }
    virtual uint32_t dma_completed_id(uint64_t /*at*/) { return 0; }
    // SNAX CSR window [960,1175]: streamer config + start + simbacore + gauges
    virtual void snax_write(uint32_t /*csr*/, uint32_t /*val*/, uint64_t /*at*/) {}
    // Does this SNAX CSR write sit on the critical path (charge offload latency), or is
    // it overlapped with accelerator compute (CSR pre-loading -> free)? Default: charge.
    virtual bool snax_write_serializes(uint32_t /*csr*/, uint64_t /*at*/) const { return true; }
    virtual uint32_t snax_read(uint32_t /*csr*/, uint64_t /*at*/, bool& is_poll) {
        is_poll = false;
        return 0;
    }
    // sync markers
    virtual void fence(int /*hart*/, uint64_t /*at*/) {}
};

struct Machine {
    Memory mem;
    std::vector<Hart> harts;
    World* world = nullptr;

    // htif
    uint32_t tohost = 0, fromhost = 0;   // symbol addresses
    bool exited = false;
    int exit_code = 0;

    // config returned by cluster CSRs
    uint32_t cluster_base_l = 0x10000000, cluster_base_h = 0;
    uint32_t core_num = 2;               // total harts in cluster
    bool quiet = false;
    // The app's own FP check_result (its "ref = N" / "N/M errors" lines) cannot pass under
    // memsim (timing + integer/layout, not the bf16/fp8 datapath), so it is suppressed by
    // default (set MEMSIM_SHOW_APP_CHECK=1 to see it). The model's correctness signal is the
    // --verify layout/BIST cross-check; see target/snitch_cluster/sim/docs/memsim.md.
    bool show_app_check = false;
    std::string app_line_;               // line accumulator for the filter
    static bool is_app_check_line(const std::string& s) {
        return s.find("ref =") != std::string::npos ||
               s.find("Checking results") != std::string::npos ||
               ((s.rfind("FAIL", 0) == 0 || s.rfind("PASS", 0) == 0) &&
                s.find("errors") != std::string::npos);
    }

    // Process a store that targeted the `tohost` address (fesvr protocol).
    void htif_tohost(uint32_t value) {
        if (value & 1) {                 // exit
            if (!app_line_.empty()) {    // flush any trailing partial line
                if (show_app_check || !is_app_check_line(app_line_))
                    std::fwrite(app_line_.data(), 1, app_line_.size(), stdout);
                app_line_.clear();
            }
            exit_code = (int)(value >> 1);
            exited = true;
            for (auto& h : harts) h.state = HartState::Halted;
            return;
        }
        // syscall: value points to a syscall_mem[8] block (uint64 each).
        uint32_t p = value;
        uint64_t magic = mem.ld64(p + 0);
        if (magic == 64) {               // sys_write(fd, buf, len)
            uint64_t fd = mem.ld64(p + 8);
            uint64_t buf = mem.ld64(p + 16);
            uint64_t len = mem.ld64(p + 24);
            std::vector<uint8_t> tmp(len);
            mem.read((uint32_t)buf, tmp.data(), (uint32_t)len);
            FILE* out = (fd == 2) ? stderr : stdout;
            if (fd == 2 || show_app_check) {
                std::fwrite(tmp.data(), 1, len, out);
                std::fflush(out);
            } else {
                // Line-buffer stdout so we can drop the app's FP check_result lines.
                for (uint32_t i = 0; i < (uint32_t)len; i++) {
                    app_line_ += (char)tmp[i];
                    if (tmp[i] == '\n') {
                        if (!is_app_check_line(app_line_))
                            std::fwrite(app_line_.data(), 1, app_line_.size(), out);
                        app_line_.clear();
                    }
                }
                std::fflush(out);
            }
        }
        // ack
        mem.st32(fromhost, 1);
        mem.st32(tohost, 0);
    }
};
