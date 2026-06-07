// Copyright 2026 KU Leuven. memsim — minimal ELF32 little-endian loader.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "mem.hpp"

struct ElfImage {
    uint32_t entry = 0;
    std::unordered_map<std::string, uint32_t> symbols;  // name -> value
    bool ok = false;
    std::string err;

    uint32_t sym(const char* name, uint32_t fallback = 0) const {
        auto it = symbols.find(name);
        return it == symbols.end() ? fallback : it->second;
    }
};

inline uint16_t rd16(const uint8_t* p) { return p[0] | (p[1] << 8); }
inline uint32_t rd32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

inline ElfImage load_elf(const char* path, Memory& mem) {
    ElfImage img;
    FILE* f = std::fopen(path, "rb");
    if (!f) { img.err = std::string("cannot open ") + path; return img; }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if ((long)std::fread(buf.data(), 1, sz, f) != sz) {
        img.err = "short read"; std::fclose(f); return img;
    }
    std::fclose(f);

    if (sz < 52 || std::memcmp(buf.data(), "\x7f""ELF", 4) != 0) {
        img.err = "not an ELF"; return img;
    }
    if (buf[4] != 1) { img.err = "not ELF32"; return img; }  // EI_CLASS=ELFCLASS32
    if (buf[5] != 1) { img.err = "not little-endian"; return img; }

    const uint8_t* e = buf.data();
    img.entry = rd32(e + 24);
    uint32_t phoff = rd32(e + 28);
    uint32_t shoff = rd32(e + 32);
    uint16_t phentsize = rd16(e + 42);
    uint16_t phnum = rd16(e + 44);
    uint16_t shentsize = rd16(e + 46);
    uint16_t shnum = rd16(e + 48);

    // Load PT_LOAD segments.
    for (uint16_t i = 0; i < phnum; i++) {
        const uint8_t* ph = e + phoff + (uint32_t)i * phentsize;
        uint32_t p_type = rd32(ph + 0);
        if (p_type != 1) continue;  // PT_LOAD
        uint32_t p_offset = rd32(ph + 4);
        uint32_t p_vaddr = rd32(ph + 8);
        uint32_t p_filesz = rd32(ph + 16);
        uint32_t p_memsz = rd32(ph + 20);
        if (p_filesz) mem.write(p_vaddr, e + p_offset, p_filesz);
        // bss tail (memsz>filesz) is already zero in the lazy memory model.
        (void)p_memsz;
    }

    // Parse section headers for .symtab + its .strtab.
    for (uint16_t i = 0; i < shnum; i++) {
        const uint8_t* sh = e + shoff + (uint32_t)i * shentsize;
        uint32_t sh_type = rd32(sh + 4);
        if (sh_type != 2) continue;  // SHT_SYMTAB
        uint32_t sh_offset = rd32(sh + 16);
        uint32_t sh_size = rd32(sh + 20);
        uint32_t sh_link = rd32(sh + 24);   // strtab section index
        uint32_t sh_entsize = rd32(sh + 36);
        const uint8_t* strsh = e + shoff + sh_link * shentsize;
        uint32_t str_off = rd32(strsh + 16);
        if (sh_entsize == 0) sh_entsize = 16;
        for (uint32_t o = 0; o + sh_entsize <= sh_size; o += sh_entsize) {
            const uint8_t* sym = e + sh_offset + o;
            uint32_t st_name = rd32(sym + 0);
            uint32_t st_value = rd32(sym + 4);
            const char* nm = (const char*)(e + str_off + st_name);
            if (nm[0]) img.symbols[std::string(nm)] = st_value;
        }
    }

    img.ok = true;
    return img;
}
