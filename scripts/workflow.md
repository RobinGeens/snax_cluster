# SimbaCore Workflow commands

## Prepare RTL

This is entirely managed by Robin.

- Make changes in chisel-ssm and push to git. This automatically updates SimbaCore.sv
- When architecture changes:
  - parse widths from SimbaCore.sv and update shell_wrapper accordingly
  - `python ./simbacore-scripts/update_simbacore_params.py`
  - Manually check the Streamer widths in config.hjson are still correct

## Build RTL/simulator

`bash scripts/build_sim.sh`

## Test

`bash scripts/regression_test.sh`

## Debug

`cd target/snitch_cluster`

[bash|target] Run SimbaCore test programs

- `bin/snitch_cluster.vsim sw/apps/nop/build/nop.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/main/build/main.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/osgemm/build/osgemm.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/isgemm/build/isgemm.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/simd/build/simd.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/fft/build/fft.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/main-full/build/main-full.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/streamer-burner/build/streamer-burner.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/core-burner/build/core-burner.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/rmsnorm/build/rmsnorm.elf | tee vsim.log`
- `bin/snitch_cluster.vsim sw/apps/suc/build/suc.elf | tee vsim.log`

[bash|target] Run SimbaCore test program in GUI (with VNC)

- `bin/snitch_cluster.vsim.gui sw/apps/main/build/main.elf`
- `bin/snitch_cluster.vsim.gui sw/apps/main-full/build/main-full.elf`
- `bin/snitch_cluster.vsim.gui sw/apps/osgemm/build/osgemm.elf`
- `bin/snitch_cluster.vsim.gui sw/apps/isgemm/build/isgemm.elf`
- `bin/snitch_cluster.vsim.gui sw/apps/fft/build/fft.elf`
- `bin/snitch_cluster.vsim.gui sw/apps/simd/build/simd.elf`
- `bin/snitch_cluster.vsim.gui sw/apps/rmsnorm/build/rmsnorm.elf`
- `bin/snitch_cluster.vsim.gui sw/apps/suc/build/suc.elf`

[snax|target] Make traces (from .dasm to .txt)

`make traces`
