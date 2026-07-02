// Copyright 2026 KU Leuven. memsim — value-carrying address space.
// See target/snitch_cluster/sim/docs/memsim.md for the overall design.
//
// A sparse, lazily-backed, zero-initialized 32-bit address space. Reads of
// never-written addresses return 0 (so .bss / fresh TCDM read as zero without
// explicit init). Little-endian. Pages are 4 KiB.
#pragma once
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

class Memory {
   public:
    static constexpr uint32_t PAGE_BITS = 12;
    static constexpr uint32_t PAGE_SIZE = 1u << PAGE_BITS;
    static constexpr uint32_t PAGE_MASK = PAGE_SIZE - 1;

    // Raw byte access -------------------------------------------------------
    uint8_t* page_for(uint32_t addr, bool create) {
        uint32_t pno = addr >> PAGE_BITS;
        auto it = pages_.find(pno);
        if (it != pages_.end()) return it->second.data();
        if (!create) return nullptr;
        auto& vec = pages_[pno];
        vec.assign(PAGE_SIZE, 0);
        return vec.data();
    }

    void write(uint32_t addr, const void* src, uint32_t n) {
        const uint8_t* s = static_cast<const uint8_t*>(src);
        while (n) {
            uint32_t off = addr & PAGE_MASK;
            uint32_t chunk = PAGE_SIZE - off;
            if (chunk > n) chunk = n;
            uint8_t* p = page_for(addr, true) + off;
            std::memcpy(p, s, chunk);
            addr += chunk;
            s += chunk;
            n -= chunk;
        }
    }

    void read(uint32_t addr, void* dst, uint32_t n) {
        uint8_t* d = static_cast<uint8_t*>(dst);
        while (n) {
            uint32_t off = addr & PAGE_MASK;
            uint32_t chunk = PAGE_SIZE - off;
            if (chunk > n) chunk = n;
            uint8_t* p = page_for(addr, false);
            if (p)
                std::memcpy(d, p + off, chunk);
            else
                std::memset(d, 0, chunk);
            addr += chunk;
            d += chunk;
            n -= chunk;
        }
    }

    // Typed little-endian helpers ------------------------------------------
    uint32_t ld32(uint32_t a) { uint32_t v; read(a, &v, 4); return v; }
    uint16_t ld16(uint32_t a) { uint16_t v; read(a, &v, 2); return v; }
    uint8_t  ld8 (uint32_t a) { uint8_t  v; read(a, &v, 1); return v; }
    uint64_t ld64(uint32_t a) { uint64_t v; read(a, &v, 8); return v; }
    void st32(uint32_t a, uint32_t v) { write(a, &v, 4); }
    void st16(uint32_t a, uint16_t v) { write(a, &v, 2); }
    void st8 (uint32_t a, uint8_t  v) { write(a, &v, 1); }
    void st64(uint32_t a, uint64_t v) { write(a, &v, 8); }

   private:
    std::unordered_map<uint32_t, std::vector<uint8_t>> pages_;
};
