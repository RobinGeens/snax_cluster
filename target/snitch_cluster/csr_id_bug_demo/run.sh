#!/usr/bin/env bash
# CSR-read id-drop bug demo (see README.md). Compiles the taped-out translator +
# testbench into the existing work-vsim library (which already holds snitch, the
# generated ReqRspManager, snax_csr_mux_demux and all packages), then vopt + vsim.
#
# Usage:  cd target/snitch_cluster && ./csr_id_bug_demo/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # -> target/snitch_cluster

ROOT=$(git rev-parse --show-toplevel)

VLOG_FLAGS=(-incr -sv -svinputport=compat -override_timescale 1ns/1ps \
  -suppress 2583 -suppress 13314 -64 -work work-vsim/ \
  +incdir+"$ROOT/hw/reqrsp_interface/include" \
  +incdir+"$ROOT/hw/snitch/include")

vlog "${VLOG_FLAGS[@]}" \
  csr_id_bug_demo/snax_intf_translator_orig.sv \
  csr_id_bug_demo/tb_snitch_csr_id_bug.sv

vopt +acc -work work-vsim tb_snitch_csr_id_bug -o tb_snitch_csr_id_bug_opt

vsim -c -64 -work work-vsim tb_snitch_csr_id_bug_opt -do "run -all; quit -f"
