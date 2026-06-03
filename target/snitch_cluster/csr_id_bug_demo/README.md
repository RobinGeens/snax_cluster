# snax_intf_translator CSR-read id-drop bug

A taped-out hardware bug can deliver a SNAX CSR read to the **wrong** destination
register. This directory reproduces the bug on the real Snitch core + real
taped-out RTL, and proves the software fix (the chip is taped out, so no HW
change is possible).

## The software fix (the instructions you run)

For any SNAX-range CSR read (`0x3c0..0x5ff`, e.g. the `R10_DELAY_GAUGE` at `0x495`),
insert `add rd,rd,zero` immediately after the `csrr`:

```c
static inline uint32_t read_snax_csr_safe(uint32_t addr) {
    uint32_t v;
    asm volatile("csrr %0, %1\n\t"
                 "add  %0, %0, zero"   // the fix
                 : "=r"(v) : "i"(addr));
    return v;
}
```

So a poll like `while (read_csr(0x495) < target)` must compile to:

```
buggy :  csrr a3,0x495 ;                     bltu a3,a2,<loop>     # HANGS
fixed :  csrr a3,0x495 ;  add a3,a3,zero ;   bltu a3,a2,<loop>     # OK
```

The `add a3,a3,zero` is the entire fix. The next section explains why it works.

## Why the bug happens, and why the filler fixes it

The translator delivers the CSR response to the GPR id **currently on the offload
bus**, which is `acc_qreq_o.id = rd` of the **decode-stage** instruction
(`snitch.sv:384`, `rd` at `:529`) — combinational, not latched. (The taped-out FIFO
that was meant to remember the id never pushes for this csrman, because the manager
answers reads combinationally, so the response is always live-passthrough.)

A wrong delivery needs both:

1. **the response is deferred ≥1 cycle** — the `snitch_cc` `stream_arbiter` withholds
   the snax response grant. Software cannot prevent this.
2. **during that defer, the decode-stage `rd` has advanced** to a different GPR.
   Software *can* prevent this.

In the buggy poll, while the `csrr a3` response is withheld the front-end advances to
`bltu a3,a2`, whose `rd` field is **x29** (the branch immediate bits). The gauge value
is delivered to x29; **a3 is never written**, its scoreboard bit never clears, and the
`bltu` waits on a3 forever → **hang**.

`add a3,a3,zero` fixes (2): it has `rd = a3` **and reads a3**, so in-order issue
**stalls** on a3's scoreboard bit and cannot advance. The bus id stays `a3` for the
whole defer, the gauge lands on a3, and the poll proceeds. One filler covers any defer
depth — issue cannot move past the stalled instruction.

> Note: "consume the result immediately" is **not** the fix. The filler must both
> (a) carry `rd == the read's destination` and (b) read it. `add rd,rd,zero` does both.

## The demonstration

`tb_snitch_csr_id_bug.sv` instantiates the real Snitch core, the real taped-out
translator (`snax_intf_translator_orig.sv` = git-HEAD logic, the version that shipped),
the real `snax_csr_mux_demux`, and the real generated `ReqRspManager`
(`readOnlyReg_2 = 0xCAFEF00D`, the gauge). Two chains run side by side on real cores —
one runs the buggy program, one runs the fixed program.

```
snitch (REAL) --acc--> snax_intf_translator_orig (REAL, taped-out, buggy)
       ^                       |
       |                  snax_csr_* (REAL snax_csr_mux_demux)
   rsp_defer  <----  ReqRspManager (REAL; readOnlyReg_2 = gauge 0xCAFEF00D)
```

Everything the SW fix controls (the offload, the decode-stage `rd`, the scoreboard stall)
is produced by the real core. The only modeled element is `rsp_defer`: a
1-cycle withhold of the response handshake, standing in for the `stream_arbiter` grant
defer. It is intentionally not a register: a registered stage would latch the id while
the `csrr` is still in decode and hide the bug; zero-latency wiring would consume the
response the same cycle and also hide it. The 1-cycle defer is exactly the window that
lets the decode-stage `rd` advance.

Address path is bit-exact to the real gauge read: `0x495` − `0x3c0` (translator) = `213`,
− `204` (mux) = `9` = csrman `readOnlyReg_2`.

## Run

```bash
cd target/snitch_cluster
./csr_id_bug_demo/run.sh
```

PASS = the buggy poll **hangs** (gauge delivered to x29, not a3) AND the fixed poll
**completes** (gauge delivered to a3) — both on the real core:

```
RESULT: PASS
```

