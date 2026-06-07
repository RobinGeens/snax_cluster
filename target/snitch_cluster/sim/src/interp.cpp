// Copyright 2026 KU Leuven. memsim — RV32IMA(+FP-as-int) interpreter.
#include "interp.hpp"

#include <cstdio>
#include <cstdlib>

#include "snax_csr.hpp"

namespace {

inline int32_t sext(uint32_t v, int bits) {
    uint32_t m = 1u << (bits - 1);
    return (int32_t)((v ^ m) - m);
}

// ---- CSR access. Returns true if this was a RO read of world/poll state. ----
bool csr_read(Machine& m, Hart& h, uint32_t csr, uint32_t& out) {
    switch (csr) {
        case CSR_MHARTID:        out = (uint32_t)h.id; return false;
        case CSR_MISA:           out = 0x40001129u;    return false;  // RV32IMAFD
        case CSR_MCYCLE: case CSR_CYCLE:   out = (uint32_t)h.cycle;        return false;
        case CSR_MCYCLEH:        out = (uint32_t)(h.cycle >> 32);          return false;
        case CSR_TIME:           out = (uint32_t)h.cycle;        return false;
        case CSR_CLUSTER_BASE_L: out = m.cluster_base_l;         return false;
        case CSR_CLUSTER_BASE_H: out = m.cluster_base_h;         return false;
        case CSR_CORE_INFO:      out = (m.core_num << 16) | (uint32_t)h.id; return false;
        default: break;
    }
    // SNAX streamer/SimbaCore CSR window routes to the timing world.
    if (csr >= SNAX_STREAMER_CFG_LO && csr <= SNAX_ISCORE_TILE_CNT) {
        m.world->advance_to(h.cycle);
        bool is_poll = false;
        out = m.world->snax_read(csr, h.cycle, is_poll);
        return is_poll;
    }
    // Generic backing store (other M/S CSRs).
    auto it = h.csr.find(csr);
    out = it == h.csr.end() ? 0 : it->second;
    return false;
}

void csr_write(Machine& m, Hart& h, uint32_t csr, uint32_t val) {
    if (csr >= SNAX_STREAMER_CFG_LO && csr <= SNAX_ISCORE_TILE_CNT) {
        m.world->snax_write(csr, val, h.cycle);
        return;
    }
    h.csr[csr] = val;
}

// Does this instruction read `reg` as a GPR source operand? Used by the load-use /
// mul-div scoreboard to charge a consumer stall (Snitch single-issue, snitch.sv:455-475).
// LUI/AUIPC/JAL have no GPR source; CSR-immediate forms (f3&4) don't read rs1; fence/ecall
// read none. R-type/branch/store/AMO read rs1+rs2; the rest read rs1.
static bool reads_reg(uint32_t op, uint32_t f3, uint32_t reg, uint32_t rs1, uint32_t rs2) {
    if (!reg) return false;
    bool r1 = false, r2 = false;
    switch (op) {
        case 0x37: case 0x17: case 0x6f: case 0x0f: break;       // LUI/AUIPC/JAL/FENCE: none
        case 0x67: case 0x03: case 0x07: case 0x13: r1 = true; break;  // JALR/LOAD/LOAD-FP/OP-IMM
        case 0x63: case 0x23: case 0x27: case 0x33: case 0x2f:   // BR/STORE/STORE-FP/OP/AMO
            r1 = true; r2 = true; break;
        case 0x73: r1 = (f3 == 1 || f3 == 2 || f3 == 3); break;  // CSRRW/S/C read rs1
        default: r1 = true; r2 = true; break;
    }
    return (r1 && rs1 == reg) || (r2 && rs2 == reg);
}

}  // namespace

Step interp_step(Machine& m, Hart& h, DmaStage& dma) {
    uint32_t pc = h.pc;
    uint32_t insn = m.mem.ld32(pc);
    uint32_t next = pc + 4;
    h.cycle += 1;  // base scalar latency; refined by the world later

    uint32_t op = insn & 0x7f;
    uint32_t rd = (insn >> 7) & 0x1f;
    uint32_t f3 = (insn >> 12) & 7;
    uint32_t rs1 = (insn >> 15) & 0x1f;
    uint32_t rs2 = (insn >> 20) & 0x1f;
    uint32_t f7 = (insn >> 25) & 0x7f;
    uint32_t a = h.x[rs1], b = h.x[rs2];

    // Load-use / mul-div consumer stall: if this instruction reads a register whose
    // load/MUL/DIV result is still in flight, advance to when it lands (snitch.sv scoreboard).
    // Independent instructions don't match -> stay 1 cc. Timing-only (value already correct).
    if (h.pend_reg && reads_reg(op, f3, h.pend_reg, rs1, rs2)) {
        if (h.pend_ready > h.cycle) h.cycle = h.pend_ready;
        h.pend_reg = 0;
    }

    Step result = Step::Normal;

    switch (op) {
        case 0x37: h.set(rd, insn & 0xfffff000u); break;             // LUI
        case 0x17: h.set(rd, pc + (insn & 0xfffff000u)); break;      // AUIPC
        case 0x6f: {                                                 // JAL
            int32_t imm = sext(((insn >> 31) & 1) << 20 | ((insn >> 12) & 0xff) << 12 |
                                   ((insn >> 20) & 1) << 11 | ((insn >> 21) & 0x3ff) << 1,
                               21);
            h.set(rd, next);
            next = pc + imm;
            break;
        }
        case 0x67: {                                                 // JALR
            int32_t imm = sext(insn >> 20, 12);
            uint32_t t = (a + imm) & ~1u;
            h.set(rd, next);
            next = t;
            break;
        }
        case 0x63: {                                                 // BRANCH
            int32_t imm = sext(((insn >> 31) & 1) << 12 | ((insn >> 7) & 1) << 11 |
                                   ((insn >> 25) & 0x3f) << 5 | ((insn >> 8) & 0xf) << 1,
                               13);
            bool take = false;
            switch (f3) {
                case 0: take = (a == b); break;                      // BEQ
                case 1: take = (a != b); break;                      // BNE
                case 4: take = ((int32_t)a < (int32_t)b); break;     // BLT
                case 5: take = ((int32_t)a >= (int32_t)b); break;    // BGE
                case 6: take = (a < b); break;                       // BLTU
                case 7: take = (a >= b); break;                      // BGEU
            }
            if (take) next = pc + imm;
            break;
        }
        case 0x03: {                                                 // LOAD
            int32_t imm = sext(insn >> 20, 12);
            uint32_t addr = a + imm;
            uint32_t v = 0;
            switch (f3) {
                case 0: v = (int32_t)(int8_t)m.mem.ld8(addr); break;   // LB
                case 1: v = (int32_t)(int16_t)m.mem.ld16(addr); break; // LH
                case 2: v = m.mem.ld32(addr); break;                   // LW
                case 4: v = m.mem.ld8(addr); break;                    // LBU
                case 5: v = m.mem.ld16(addr); break;                   // LHU
            }
            h.set(rd, v);
            if (rd) { h.pend_reg = rd; h.pend_ready = h.cycle + LOAD_USE_STALL; }  // load-use producer
            break;
        }
        case 0x23: {                                                 // STORE
            int32_t imm = sext(((insn >> 25) << 5) | ((insn >> 7) & 0x1f), 12);
            uint32_t addr = a + imm;
            switch (f3) {
                case 0: m.mem.st8(addr, (uint8_t)b); break;          // SB
                case 1: m.mem.st16(addr, (uint16_t)b); break;        // SH
                case 2: m.mem.st32(addr, b); break;                  // SW
            }
            if (addr == m.tohost) {                                  // htif
                m.htif_tohost(b);
                if (m.exited) result = Step::Halt;
            }
            break;
        }
        case 0x13: {                                                 // OP-IMM
            int32_t imm = sext(insn >> 20, 12);
            uint32_t sh = rs2;  // shamt
            switch (f3) {
                case 0: h.set(rd, a + imm); break;                       // ADDI
                case 2: h.set(rd, (int32_t)a < imm ? 1 : 0); break;      // SLTI
                case 3: h.set(rd, a < (uint32_t)imm ? 1 : 0); break;     // SLTIU
                case 4: h.set(rd, a ^ imm); break;                       // XORI
                case 6: h.set(rd, a | imm); break;                       // ORI
                case 7: h.set(rd, a & imm); break;                       // ANDI
                case 1: h.set(rd, a << sh); break;                       // SLLI
                case 5: h.set(rd, (f7 & 0x20) ? (uint32_t)((int32_t)a >> sh)
                                              : (a >> sh)); break;       // SRLI/SRAI
            }
            break;
        }
        case 0x33: {                                                 // OP
            if (f7 == 1) {                                           // RV32M
                int32_t sa = (int32_t)a, sb = (int32_t)b;
                switch (f3) {
                    case 0: h.set(rd, a * b); break;                                  // MUL
                    case 1: h.set(rd, (uint32_t)(((int64_t)sa * (int64_t)sb) >> 32)); break;       // MULH
                    case 2: h.set(rd, (uint32_t)(((int64_t)sa * (int64_t)(uint64_t)b) >> 32)); break; // MULHSU
                    case 3: h.set(rd, (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32)); break;       // MULHU
                    case 4: h.set(rd, sb == 0 ? 0xffffffffu : (uint32_t)(sa == INT32_MIN && sb == -1 ? sa : sa / sb)); break; // DIV
                    case 5: h.set(rd, b == 0 ? 0xffffffffu : a / b); break;           // DIVU
                    case 6: h.set(rd, sb == 0 ? a : (uint32_t)(sa == INT32_MIN && sb == -1 ? 0 : sa % sb)); break; // REM
                    case 7: h.set(rd, b == 0 ? a : a % b); break;                     // REMU
                }
                // MUL/DIV offloaded to the shared muldiv -> consumer of rd stalls (snitch.sv:1086)
                if (rd) { h.pend_reg = rd; h.pend_ready = h.cycle + (f3 < 4 ? MUL_STALL : DIV_STALL); }
            } else {
                uint32_t sh = b & 0x1f;
                switch (f3) {
                    case 0: h.set(rd, (f7 & 0x20) ? a - b : a + b); break;            // ADD/SUB
                    case 1: h.set(rd, a << sh); break;                               // SLL
                    case 2: h.set(rd, (int32_t)a < (int32_t)b ? 1 : 0); break;        // SLT
                    case 3: h.set(rd, a < b ? 1 : 0); break;                          // SLTU
                    case 4: h.set(rd, a ^ b); break;                                  // XOR
                    case 5: h.set(rd, (f7 & 0x20) ? (uint32_t)((int32_t)a >> sh) : (a >> sh)); break; // SRL/SRA
                    case 6: h.set(rd, a | b); break;                                  // OR
                    case 7: h.set(rd, a & b); break;                                  // AND
                }
            }
            break;
        }
        case 0x0f:                                                  // MISC-MEM (fence/fence.i)
            m.world->fence(h.id, h.cycle);
            break;
        case 0x2f: {                                                // AMO (.w only, f3==2)
            uint32_t f5 = f7 >> 2;
            uint32_t addr = a;
            uint32_t old = m.mem.ld32(addr);
            uint32_t res = old;
            switch (f5) {
                case 0x00: res = old + b; break;                    // amoadd.w
                case 0x01: res = b; break;                          // amoswap.w
                case 0x04: res = old ^ b; break;                    // amoxor.w
                case 0x08: res = old | b; break;                    // amoor.w
                case 0x0c: res = old & b; break;                    // amoand.w
                case 0x10: res = (int32_t)old < (int32_t)b ? b : old; break; // amomin.w (note: min)
                case 0x14: res = (int32_t)old > (int32_t)b ? b : old; break; // amomax.w
                case 0x18: res = old < b ? b : old; break;          // amominu.w
                case 0x1c: res = old > b ? b : old; break;          // amomaxu.w
                case 0x02: h.set(rd, old); /* lr.w */ break;        // lr.w: no write
                case 0x03: m.mem.st32(addr, b); h.set(rd, 0); /* sc.w success */ break;
            }
            if (f5 != 0x02 && f5 != 0x03) {
                m.mem.st32(addr, res);
                h.set(rd, old);
            }
            break;
        }
        case 0x73: {                                                // SYSTEM
            if (f3 == 0) {
                uint32_t fn = insn >> 20;
                if (fn == 0x105) { result = Step::Wfi; break; }      // wfi
                if (fn == 0x302) { /* mret */ break; }
                // ecall/ebreak: ignore (not used by these apps)
                break;
            }
            // CSR ops
            uint32_t csr = insn >> 20;
            if (csr == CSR_HW_BARRIER) {                             // csrr x0,0x7C2
                h.pc = next;
                h.state = HartState::AtBarrier;
                return Step::Barrier;
            }
            uint32_t old = 0;
            bool isPoll = csr_read(m, h, csr, old);
            uint32_t src = (f3 & 4) ? rs1 /* zimm */ : a;
            uint32_t neu = old;
            switch (f3 & 3) {
                case 1: neu = src; break;                           // CSRRW(I)
                case 2: neu = old | src; break;                     // CSRRS(I)
                case 3: neu = old & ~src; break;                    // CSRRC(I)
            }
            if (!(((f3 & 3) != 1) && rs1 == 0)) {                   // skip write if RS1=x0 for set/clr
                csr_write(m, h, csr, neu);
                if (csr >= SNAX_STREAMER_CFG_LO && csr <= SNAX_ISCORE_TILE_CNT &&
                    m.world->snax_write_serializes(csr, h.cycle))
                    h.cycle += SNAX_CSR_WRITE_COST;                 // offload latency, unless
                                                                    // overlapped (snax_write_serializes)
            }
            h.set(rd, old);
            if (isPoll) result = Step::PollRead;
            break;
        }
        case 0x07: {                                                // LOAD-FP
            int32_t imm = sext(insn >> 20, 12);
            uint32_t addr = a + imm;
            if (f3 == 2) h.f[rd] = m.mem.ld32(addr);                // flw
            else if (f3 == 3) h.f[rd] = m.mem.ld64(addr);           // fld
            break;
        }
        case 0x27: {                                                // STORE-FP
            int32_t imm = sext(((insn >> 25) << 5) | ((insn >> 7) & 0x1f), 12);
            uint32_t addr = a + imm;
            if (f3 == 2) m.mem.st32(addr, (uint32_t)h.f[rs2]);      // fsw
            else if (f3 == 3) m.mem.st64(addr, h.f[rs2]);           // fsd
            break;
        }
        case 0x53: {                                                // OP-FP
            // Only fcvt.d.w / fcvt.s.w are implemented (sufficient for crt0); the
            // datapath model reinterprets FP bytes as integer, so other OP-FP -> 0.
            if (f7 == 0x69 || f7 == 0x68)                           // fcvt.d.w / fcvt.s.w
                h.f[rd] = (uint64_t)(int64_t)(int32_t)a;
            else
                h.f[rd] = 0;
            break;
        }
        case 0x2b:                                                  // xDMA custom-0
            switch (f7) {
                case 0x00:  // dmsrc rs2,rs1
                    dma.src = ((uint64_t)b << 32) | a; break;
                case 0x01:  // dmdst rs2,rs1
                    dma.dst = ((uint64_t)b << 32) | a; break;
                case 0x06:  // dmstr rs2(dst),rs1(src)
                    dma.dst_stride = b; dma.src_stride = a; break;
                case 0x07:  // dmrep rs1
                    dma.repeat = a; break;
                case 0x02: {  // dmcpyi rd, rs1(size), imm(rs2 field)=cfg
                    DmaDesc d;
                    d.src = dma.src; d.dst = dma.dst; d.size = a;
                    d.dst_stride = dma.dst_stride; d.src_stride = dma.src_stride;
                    d.repeat = dma.repeat; d.is_2d = (rs2 >> 1) & 1;
                    d.src_is_l3 = (d.src >> 31) & 1; d.dst_is_l3 = (d.dst >> 31) & 1;
                    d.hart = h.id;
                    uint32_t txid = m.world->dma_submit(d, h.cycle);
                    h.set(rd, txid);
                    dma.repeat = 1;  // reset like fresh config
                    break;
                }
                case 0x04: {  // dmstati rd, imm(rs2 field)
                    uint32_t v = 0;
                    if (rs2 == 0) v = m.world->dma_completed_id(h.cycle);
                    else if (rs2 == 2) v = m.world->dma_busy(h.cycle);
                    // rs2==1/3: tracking markers, rd is x0 -> no-op
                    h.set(rd, v);
                    if (rd != 0) result = Step::PollRead;
                    break;
                }
                default: break;
            }
            break;
        default:
            std::fprintf(stderr, "memsim: illegal insn 0x%08x at pc 0x%08x (hart %d)\n",
                         insn, pc, h.id);
            std::exit(2);
    }

    h.pc = next;
    // backward-branch / poll bookkeeping (used by the scheduler for wait loops)
    if (result == Step::PollRead && next <= pc) {
        if (h.last_backjump_pc == next) h.poll_spins++;
        else { h.last_backjump_pc = next; h.poll_spins = 1; }
    }
    return result;
}
