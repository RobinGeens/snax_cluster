// Copyright 2026 KU Leuven. memsim — float decode for the FP32 functional datapath.
// The accelerator's inType is FP8_ALT (e5m2: 1 sign / 5 exp / 2 mantissa, bias 15;
// 1 byte, 8 per 64-bit TCDM bank) and accType is BF16 (e8m7 = top 16 bits of fp32;
// 2 bytes, 4 per bank). The functional model stores the real bytes and computes in
// FP32, up-converting on read with these decoders. Definitions:
// chisel-ssm/chisel-float/src/main/scala/fp_unit/DataType.scala.
#pragma once
#include <cstdint>
#include <cstring>

// e5m2 byte -> float. exp=0: subnormal; exp=31: inf/nan (FP8_ALT saturates in HW,
// but a decoded inf only appears on already-out-of-range data, which the generous
// comparison tolerance handles).
inline float fp8_alt_to_f32(uint8_t b) {
    int sign = (b >> 7) & 1;
    int exp  = (b >> 2) & 0x1f;
    int mant = b & 0x3;
    float m, v;
    if (exp == 0) {                         // subnormal: (mant/4) * 2^(1-15)
        v = (mant / 4.0f) * 0.00006103515625f;          // 2^-14
    } else if (exp == 0x1f) {               // inf / nan
        v = mant ? __builtin_nanf("") : __builtin_huge_valf();
    } else {                                // normal: (1 + mant/4) * 2^(exp-15)
        m = 1.0f + mant / 4.0f;
        int e = exp - 15;
        v = m * (e >= 0 ? (float)(1u << e) : 1.0f / (float)(1u << (-e)));
    }
    return sign ? -v : v;
}

// bf16 (16-bit) -> float: it is the upper 16 bits of an IEEE fp32.
inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
