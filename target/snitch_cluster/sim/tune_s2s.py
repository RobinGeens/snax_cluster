#!/usr/bin/env python3
"""Tune an app's safe_to_start_r11 from the memsim stale-free optimal.

Runs the memsim on the app's built ELF, parses the per-shape R11 boundary from the
safe-to-start sweep, and writes optimal * MARGIN into the app's params_in.hjson.
Rebuild the app afterwards (podman) so datagen bakes the new value into data.h.

Usage (from target/snitch_cluster):
    $POD make -C sw/apps/<app>          # build with current params
    python3 sim/tune_s2s.py <app>       # host: memsim sweep -> params_in.hjson
    $POD make -C sw/apps/<app>          # rebuild with the tuned value
"""
import math
import pathlib
import re
import subprocess
import sys

MARGIN = 1.02

if len(sys.argv) != 2:
    sys.exit(__doc__)
app = sys.argv[1].rstrip("/").split("/")[-1]

tc = pathlib.Path(__file__).resolve().parents[1]
elf = tc / "sw" / "apps" / app / "build" / f"{app}.elf"
params = tc / "sw" / "apps" / app / "data" / "params_in.hjson"
if not elf.exists():
    sys.exit(f"{elf} not found - build the app first")

res = subprocess.run(
    [str(tc / "bin" / "snitch_cluster.memsim"), str(elf)],
    capture_output=True, text=True,
)
out = res.stdout + res.stderr
m = re.search(r"R11 \(y\) optimal=(\d+)/(\d+)", out)
if not m:
    sys.exit("memsim printed no R11 safe-to-start sweep (not a P2 app, or memsim failed)")
opt, total = int(m.group(1)), int(m.group(2))
val = min(math.ceil(opt * MARGIN), total)

txt = params.read_text()
new, n = re.subn(r"(safe_to_start_r11:\s*)-?\d+", rf"\g<1>{val}", txt)
if n != 1:
    sys.exit(f"expected exactly one safe_to_start_r11 key in {params}, found {n}")
params.write_text(new)
print(f"{app}: safe_to_start_r11 = {val} (memsim optimal {opt}/{total}, margin x{MARGIN})")
print("Rebuild the app to bake it into data.h")
