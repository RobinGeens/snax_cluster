# memsim

Functional + cycle-accurate **memory-transfer** simulator for the SNAX cluster +
chisel-ssm SimbaCore accelerator. Runs a compiled app `.elf` (no podman, no
license) and predicts cycle counts orders of magnitude faster than RTL `vsim`.

**Design, calibration, and validation live in
[`docs/memsim.md`](docs/memsim.md)** — not here.

## Build & run
```bash
make -C target/snitch_cluster/sim                                 # host g++ (gcc-11.2.0), seconds
./bin/snitch_cluster.memsim sw/apps/<app>/build/<app>.elf         # drop-in for ./bin/snitch_cluster.vsim
                                                                  #   runs the integer layout/BIST check
./bin/snitch_cluster.memsim sw/apps/<app>/build/<app>.elf --timing-only  # skip the check (pure timing)
```
The model does timing + integer/layout, not the bf16/fp8 datapath, so it does not produce
the apps' FP output buffers. The app's own `check_result` (its `ref = N` / `N/M errors`
lines) therefore cannot pass under memsim and is suppressed by default; it is never the
model's verdict. Correctness comes from the layout/BIST cross-check, which always runs.
Verdict: `[SUCCESS]` iff that passes and the program completed (a deadlock fails).

Env vars: `MEMSIM_DEBUG` (run summary), `MEMSIM_ACC` (per-invocation accelerator durations),
`MEMSIM_DMA` (first DMA transfers), `MEMSIM_SHOW_APP_CHECK` (un-suppress the app's FP check),
`MEMSIM_DATAPATH` (FP32 datapath vs goldens, non-tiled `main`), `MEMSIM_LAYOUT_FAULT` (audit
liveness self-test). The safe-to-start sweep prints on every P2 run (no flag).

## Calibrate DMA constants from a real trace
```bash
python3 target/snitch_cluster/sim/calib/calibrate_dma.py logs/trace_chip_00_dma_00001.log
```

## Layout
- `src/` — `main`, `interp` (RV32IMA+FP), `sched` (dual-hart co-sim), `world` (timing +
  functional datapath + verification), `cyc` (per-cycle fabric/streamers), `mem`, `elf`,
  `machine`, `hart`, `snax_csr`, `fp`.
- `test/` — `suc_grid_test.cpp` (standalone bank-conflict model test; see its header to build).
- `calib/` — DMA calibration script + `constants.json`.
