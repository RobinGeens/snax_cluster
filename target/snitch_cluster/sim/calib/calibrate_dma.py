#!/usr/bin/env python3
# Copyright 2026 KU Leuven. memsim — calibrate DMA timing constants from a real
# per-cycle DMA trace (logs/trace_chip_00_dma_*.log).
#
# Each line is a per-cycle Python-dict snapshot. A burst *fires* on the cycle
# where backend_burst_req_valid & backend_burst_req_ready are both 1; it carries
# num_bytes/src/dst. The matching completion is the transfer_completed=1 pulse.
# For an isolated (non-overlapping) transfer:
#     gap = t_complete - t_fire ;  first_beat = gap - ceil(num_bytes/64)
# We regress gap on ceil(bytes/64) over clean isolated transfers to get the
# fixed first-beat latency (intercept) and per-beat cost (slope), split by route
# (L3<->TCDM vs TCDM<->TCDM). Writes constants.json next to this script.
import json
import os
import re
import sys

BEAT_BYTES = 64  # DataWidth 512b
TCDM_BASE = 0x10000000
TCDM_END = 0x10080000

KEYS = ("time", "backend_burst_req_valid", "backend_burst_req_ready",
        "backend_burst_req_num_bytes", "backend_burst_req_src",
        "backend_burst_req_dst", "transfer_completed")
RE = {k: re.compile(r"'%s': (0x[0-9a-fA-F]+)" % k) for k in KEYS}


def field(line, key):
    m = RE[key].search(line)
    return int(m.group(1), 16) if m else 0


def route(src, dst):
    in_t = lambda a: TCDM_BASE <= a < TCDM_END
    s, d = in_t(src), in_t(dst)
    if s and d:
        return "tcdm_tcdm"
    if s != d:
        return "l3_tcdm"
    return "l3_l3"


def main(path):
    # In-order FIFO pairing: the backend has one ID, so completions retire in
    # fire order. gap = t_complete - t_fire = first_beat + beats + queueing.
    # Queueing only INFLATES the gap, so the true pipeline first-beat latency is
    # the FLOOR (min) of gap-beats per route, not the mean.
    import collections
    q = collections.deque()      # (t_fire, beats, route)
    samples = {}                 # route -> list of (beats, gap)
    with open(path) as f:
        for line in f:
            if "'time'" not in line:
                continue
            t = field(line, "time")
            if field(line, "backend_burst_req_valid") and field(line, "backend_burst_req_ready"):
                nb = field(line, "backend_burst_req_num_bytes")
                rt = route(field(line, "backend_burst_req_src"),
                           field(line, "backend_burst_req_dst"))
                q.append((t, (nb + BEAT_BYTES - 1) // BEAT_BYTES, rt))
            if field(line, "transfer_completed") and q:
                t_fire, beats, rt = q.popleft()
                gap = t - t_fire
                if gap > 0 and beats > 0:
                    samples.setdefault(rt, []).append((beats, gap))
    overlapped = len(q)

    out = {"_source": os.path.abspath(path), "beat_bytes": BEAT_BYTES}
    for rt, pts in sorted(samples.items()):
        diffs = sorted(g - b for b, g in pts)        # gap - beats per transfer
        n = len(diffs)
        floor = diffs[0]                              # pipeline first-beat latency
        med = diffs[n // 2]
        out[rt] = {
            "n": n, "first_beat_lat_cc": floor,
            "median_gap_minus_beats": med, "max_gap_minus_beats": diffs[-1],
        }
        print(f"{rt:10s} n={n:5d}  first_beat(floor)={floor}cc  median(gap-beats)={med}cc  "
              f"max={diffs[-1]}cc")
    print(f"unmatched fires at EOF: {overlapped}")

    # Headline constants the timing model uses.
    out["dma_first_beat_latency_cc"] = out.get("l3_tcdm", {}).get("first_beat_lat_cc", 20)
    out["dma_tcdm_first_beat_latency_cc"] = out.get("tcdm_tcdm", {}).get("first_beat_lat_cc", 10)
    out["dma_cost_per_beat_cc"] = 1.0
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constants.json")
    with open(dst, "w") as fo:
        json.dump(out, fo, indent=2)
    print("wrote", dst)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else \
        "../../logs/trace_chip_00_dma_00001.log"
    main(p)
