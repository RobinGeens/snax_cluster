#!/usr/bin/env python3
"""Minimal FIFO sweep. Edit the hard-coded CONFIGS list below."""

import os
import re
import shlex
import subprocess
from tabulate import tabulate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG = os.path.join(ROOT, "target", "snitch_cluster", "cfg", "snax_simbacore_cluster.hjson")
VSIM_DIR = os.path.join(ROOT, "target", "snitch_cluster")
OUT_DIR = os.path.join(ROOT, "fifo_sweep")
SUMMARY_PATH = os.path.join(OUT_DIR, "fifo_sweep_summary.txt")
BUILD_SCRIPT = os.path.join(ROOT, "scripts", "build_sim_minimal.sh")
PROGRAM_BIN = os.path.join(VSIM_DIR, "bin", "snitch_cluster.vsim")
PROGRAM_ELF = os.path.join(VSIM_DIR, "sw", "apps", "snax-simbacore-main", "build", "snax-simbacore-main.elf")

CONFIGS = [
    # 0  1  2  3  4  5  6  7  8  9  10 11 12 13 # 0  1  2  3
    ([9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 8]),
    ([9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 7]),
]


def patch_fifo_depths(reader, writer):
    txt = open(CFG).read()
    pat = re.compile(r"(fifo_depth:\s*)\[[^\]]*\]")
    seen = 0

    def repl(m):
        nonlocal seen
        seen += 1
        if seen == 1:
            vals = ", ".join(str(v) for v in reader)
            return f"{m.group(1)}[ {vals} ]"
        elif seen == 2:
            vals = ", ".join(str(v) for v in writer)
            return f"{m.group(1)}[ {vals} ]"
        return m.group(0)

    txt = pat.sub(repl, txt)
    open(CFG, "w").write(txt)


def run_build_and_sim(build_log_path, sim_log_path):
    subprocess.check_call(
        f"bash {shlex.quote(BUILD_SCRIPT)} > {shlex.quote(build_log_path)} 2>&1", cwd=ROOT, shell=True
    )
    subprocess.check_call(f"{PROGRAM_BIN} {PROGRAM_ELF} > {shlex.quote(sim_log_path)} 2>&1", cwd=VSIM_DIR, shell=True)


def parse_cycles(log_path):
    if not os.path.exists(log_path):
        raise RuntimeError(f"Log file not found: {log_path}")
    s = open(log_path).read()
    phase1_match = re.search(r"Simbacore Phase1 took\s+(\d+)\s+cycles", s)
    phase2_match = re.search(r"Simbacore Phase2 took\s+(\d+)\s+cycles", s)
    if phase1_match and phase2_match:
        return int(phase1_match.group(1)), int(phase2_match.group(1))
    nums = [int(m.group(1)) for m in re.finditer(r"Simbacore elapsed time:\s+(\d+)\s+cycles", s)]
    if len(nums) >= 2:
        return nums[-2], nums[-1]
    all_cycles = [int(m.group(1)) for m in re.finditer(r"(\d+)\s+cycles", s)]
    if len(all_cycles) >= 2:
        return all_cycles[-2], all_cycles[-1]
    raise RuntimeError(f"Could not parse cycle counts from {log_path}")


def get_config_name(reader: list[int], writer: list[int]):
    def format_list(values):
        result = []
        for v in values:
            if v >= 10 and result:  # Add separator before values >= 10 (except first)
                result.append("_")
            result.append(str(v))
        return "".join(result)

    reader_str = format_list(reader)
    writer_str = format_list(writer)
    return f"{reader_str}-{writer_str}"


def parse_summary_line(line):
    """Parse a line from summary file, return (reader, writer, phase1, phase2, total_delay, fifo_cost) or None."""
    line = line.rstrip()
    if line.startswith("-") and not any(c.isalnum() for c in line) or not line or line.startswith("reader_fifo_depth"):
        return None
    if "|" in line:
        parts = [p.strip() for p in line.split("|") if p.strip()]
    elif "\t" in line:
        parts = line.split("\t")
    else:
        parts = [p for p in line.split("  ") if p.strip()]
    if len(parts) < 2:
        return None
    idx_offset = 1 if len(parts) >= 4 and parts[0].isdigit() else 0
    reader_str = parts[idx_offset].replace(",", " ")
    writer_str = parts[idx_offset + 1].replace(",", " ")
    try:
        reader = tuple(map(int, reader_str.split()))
        writer = tuple(map(int, writer_str.split()))
        phase1 = int(parts[idx_offset + 2]) if len(parts) > idx_offset + 2 else None
        phase2 = int(parts[idx_offset + 3]) if len(parts) > idx_offset + 3 else None
        total_delay = (
            int(parts[idx_offset + 4])
            if len(parts) > idx_offset + 4
            else (phase1 + phase2 if phase1 is not None and phase2 is not None else None)
        )
        fifo_cost = int(parts[idx_offset + 5]) if len(parts) > idx_offset + 5 else sum(reader) + sum(writer)
        return (reader, writer, phase1, phase2, total_delay, fifo_cost)
    except (ValueError, IndexError):
        return None


def get_completed_configs():
    completed = set()
    if not os.path.exists(SUMMARY_PATH):
        return completed
    with open(SUMMARY_PATH, "r") as f:
        for line in f:
            parsed = parse_summary_line(line)
            if parsed:
                completed.add((parsed[0], parsed[1]))
    return completed


def read_existing_summary():
    rows = []
    if not os.path.exists(SUMMARY_PATH):
        return rows
    with open(SUMMARY_PATH, "r") as f:
        for line in f:
            parsed = parse_summary_line(line)
            if parsed and parsed[2] is not None and parsed[3] is not None:
                total_delay = parsed[4] if parsed[4] is not None else parsed[2] + parsed[3]
                fifo_cost = parsed[5] if len(parsed) > 5 and parsed[5] is not None else sum(parsed[0]) + sum(parsed[1])
                rows.append(
                    [
                        ",".join(map(str, parsed[0])),
                        ",".join(map(str, parsed[1])),
                        str(parsed[2]),
                        str(parsed[3]),
                        str(total_delay),
                        str(fifo_cost),
                    ]
                )
    return rows


def write_summary(rows):
    headers = ["reader_fifo_depth", "writer_fifo_depth", "phase1", "phase2", "total_delay", "total_fifo_cost"]
    with open(SUMMARY_PATH, "w") as f:
        f.write(tabulate(rows, headers=headers, tablefmt="simple", numalign="right", stralign="left") + "\n")


def read_and_sort_summary():
    """Read summary, sort by total FIFO cost (sum of all FIFO depths), and write back to file."""
    rows = []
    if not os.path.exists(SUMMARY_PATH):
        return rows
    with open(SUMMARY_PATH, "r") as f:
        for line in f:
            parsed = parse_summary_line(line)
            if parsed and parsed[2] is not None and parsed[3] is not None:
                reader, writer, phase1, phase2, total_delay, fifo_cost = parsed
                total_delay = (
                    total_delay
                    if total_delay is not None
                    else (phase1 + phase2 if phase1 is not None and phase2 is not None else 0)
                )
                fifo_cost = fifo_cost if fifo_cost is not None else sum(reader) + sum(writer)
                rows.append(
                    {
                        "reader": reader,
                        "writer": writer,
                        "phase1": phase1,
                        "phase2": phase2,
                        "total_delay": total_delay,
                        "fifo_cost": fifo_cost,
                        "row": [
                            ",".join(map(str, reader)),
                            ",".join(map(str, writer)),
                            str(phase1),
                            str(phase2),
                            str(total_delay),
                            str(fifo_cost),
                        ],
                    }
                )
    sorted_rows = sorted(rows, key=lambda x: x["fifo_cost"])
    # Write sorted summary back to file
    if sorted_rows:
        write_rows = [row["row"] for row in sorted_rows]
        write_summary(write_rows)
    return sorted_rows


def process_single_config(reader, writer):
    """
    Process a single FIFO config. Returns the sum of phase1 and phase2 latencies.
    If the config has already been run, parses it from the summary.
    Otherwise, runs the simulation and updates the summary.
    """
    config_name = get_config_name(reader, writer)
    reader_tuple = tuple(reader)
    writer_tuple = tuple(writer)

    # Check if config is already in summary
    completed = get_completed_configs()
    if (reader_tuple, writer_tuple) in completed:
        print(f"{config_name} (SKIPPED - already logged)", flush=True)
        # Parse from existing summary
        with open(SUMMARY_PATH, "r") as f:
            for line in f:
                parsed = parse_summary_line(line)
                if parsed and parsed[0] == reader_tuple and parsed[1] == writer_tuple:
                    if parsed[2] is not None and parsed[3] is not None:
                        total_delay = parsed[4] if parsed[4] is not None else parsed[2] + parsed[3]
                        return total_delay
        # If not found in summary (shouldn't happen), raise error
        raise RuntimeError(f"Config {config_name} marked as completed but not found in summary")

    # Config not completed - run simulation
    print(f"{config_name} (RUNNING)", flush=True)

    # Save original config
    orig = open(CFG).read()
    try:
        # Patch config file
        patch_fifo_depths(reader, writer)

        # Run build and simulation
        build_log_path = os.path.join(OUT_DIR, f"build_{config_name}.log")
        sim_log_path = os.path.join(OUT_DIR, f"vsim_{config_name}.log")
        run_build_and_sim(build_log_path, sim_log_path)

        # Parse results
        p1, p2 = parse_cycles(sim_log_path)
        print(f"  Result: phase1={p1}, phase2={p2}", flush=True)

        # Update summary: add new entry
        existing_rows = read_existing_summary()
        total_delay = p1 + p2
        fifo_cost = sum(reader) + sum(writer)
        existing_rows.append(
            [",".join(map(str, reader)), ",".join(map(str, writer)), str(p1), str(p2), str(total_delay), str(fifo_cost)]
        )
        write_summary(existing_rows)

        return p1 + p2
    finally:
        # Always restore config file
        open(CFG, "w").write(orig)


def execute_sweep():
    os.makedirs(OUT_DIR, exist_ok=True)

    for r, w in CONFIGS:
        try:
            process_single_config(r, w)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            print(f"  Skipping this config and continuing...", flush=True)
            continue

    print(f"Summary: {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    read_and_sort_summary()
    raise SystemExit(execute_sweep())
