// Focused unit test for the FAITHFUL (no-W_bc) SUC duration model.
// Sweeps the (bc_pad x dInner_tile) grid using R7's REAL captured strides to
// show the bank conflict is governed by bc_pad (bank count), NOT by dInner_tile.
//   build: g++ -std=c++17 -O2 -Isrc src/cyc.cpp test/suc_grid_test.cpp -o /tmp/suc_grid
#include <cstdio>
#include "machine.hpp"
#include "cyc.hpp"

static Agu make_r7(int dInner_tile, bool pad4) {
    Agu a;
    a.base = 0x10000000;          // arbitrary TCDM base
    a.n_s = 2;                    // 2D spatial, 4 lanes
    a.s_stride[0] = pad4 ? 160 : 128;
    a.s_stride[1] = pad4 ? 320 : 256;
    // temporal nest captured from the real R7: [4,16,12,(D/delaySU)] / [512,8,2432,0]
    a.t_bound[0] = 4;   a.t_stride[0] = 512;
    a.t_bound[1] = 16;  a.t_stride[1] = 8;
    a.t_bound[2] = 12;  a.t_stride[2] = 2432;
    a.t_bound[3] = dInner_tile / 4;  a.t_stride[3] = 0;   // reuse dim = D/delaySU
    a.enabled = true;
    return a;
}

int main() {
    const int seqLen = 192;
    printf("FAITHFUL SUC model: suc cycles and conflict factor vs pure compute (seqLen*dInner_tile)\n");
    printf("%-8s %-6s %10s %10s %8s\n", "dInner", "bc_pad", "compute", "suc", "factor");
    // The emergent, falsifiable property: 4-bank R7 (bc_pad=4) -> ~1.0x; 2-bank (bc_pad=0)
    // -> ~1.75x, NOT the group-synchronous worst case 2.0x. The 1.75x falls out of the
    // per-lane address-FIFO read-ahead + the +1-bank dim1 shift between refreshes (lanes on
    // free banks pipeline into the next refresh while a conflicted lane finishes the current
    // one). FLAT across dInner_tile, and NO efficiency factor anywhere. Matches vsim 1.756x.
    int fails = 0;
    for (int dInner : {24, 48, 96}) {
        for (int pad4 : {0, 1}) {
            Agu a = make_r7(dInner, pad4);
            uint64_t suc = cyc_suc_duration(a, seqLen, dInner, 0);
            long compute = (long)seqLen * dInner;
            double factor = (double)suc / compute;
            double want = pad4 ? 1.0 : 1.75;
            bool ok = factor >= want - 0.02 && factor <= want + 0.04;   // +startup-fill tolerance
            if (!ok) fails++;
            printf("%-8d %-6d %10ld %10llu %8.3f  %s\n",
                   dInner, pad4 ? 4 : 0, compute,
                   (unsigned long long)suc, factor, ok ? "ok" : "FAIL");
        }
    }
    printf(fails ? "\nFAIL: %d cases off the bank-count target\n"
                 : "\nPASS: conflict factor = bank count, flat across dInner_tile\n", fails);
    return fails ? 1 : 0;
}
