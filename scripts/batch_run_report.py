#!/usr/bin/env python3
"""Persistent live report for batch runs (see batch_run.py).

There is ONE report file, accumulated over time across batch runs, at the repo root:
    <root>/report.json   machine-readable, keyed by job id (app__tag)
    <root>/report.md     rendered Markdown table

Each batch run writes its own logs/elfs to <root>/batch_run_out/<timestamp>/, and
merges its results into the single report. A row records which batch run (timestamp)
the stored number came from. The merge is self-cleaning PER APP: a batch run wipes
every existing row for the apps it runs and writes only that batch run's jobs, so
reruns replace stale/contaminated/orphan rows. Apps the batch run doesn't touch are
kept, so results still accumulate across batch runs by app.

Usage (run from the repo root, or pass the dir holding report.json):
    python3 scripts/batch_run_report.py                 # live-watch the report
    python3 scripts/batch_run_report.py --once          # render once
    python3 scripts/batch_run_report.py --merge batch_run_out/<ts>   # import a run folder
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

# Markers emitted by the simulator (same ones the regression test parses).
RE_ERRORS = re.compile(r"Finished with exit code\s+(\d+)")
RE_ERRORS_ALT = re.compile(r"Errors:\s+(\d+)")
RE_SIMBACORE = re.compile(r"Simbacore elapsed time:\s+(\d+)\s+cycles")
RE_TOTAL = re.compile(r"Snitch elapsed time:\s+(\d+)\s+cycles")

REPORT_JSON = "report.json"
REPORT_MD = "report.md"

# Status -> emoji. All chosen to render as double-width glyphs so the raw-text
# table stays aligned in a terminal (markdown viewers ignore the padding).
EMOJI = {
    "PASS": "✅", "FAIL": "❌", "RUNNING": "🟢", "BUILDING": "🔨",
    "BUILD_FAIL": "🧱", "TIMEOUT": "🕒", "QUEUED": "🟡", "NO_RESULT": "❔",
}
_WIDE = set(EMOJI.values())


def _dw(s):
    """Display width: the emoji above occupy two columns but len() counts one."""
    return len(s) + sum(1 for c in s if c in _WIDE)


def _pad(s, width, right=False):
    fill = " " * max(0, width - _dw(s))
    return fill + s if right else s + fill


def parse_log(path):
    """Return (errors, simbacore, total) strings from a run log, or Nones."""
    if not path or not os.path.exists(path):
        return None, None, None
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return None, None, None
    m = RE_ERRORS.findall(text) or RE_ERRORS_ALT.findall(text)
    sc = RE_SIMBACORE.findall(text)
    tot = RE_TOTAL.findall(text)
    return (m[-1] if m else None), (sc[-1] if sc else None), (tot[-1] if tot else None)


def _display_status(state, errors):
    if state == "queued":
        return "QUEUED"
    if state == "building":
        return "BUILDING"
    if state == "running":
        return "RUNNING"
    if state == "build_failed":
        return "BUILD_FAIL"
    if state == "timeout":
        return "TIMEOUT"
    if state == "done":
        if errors is None:
            return "NO_RESULT"
        return "PASS" if errors == "0" else "FAIL"
    return state.upper()


def _read_json(path):
    for _ in range(5):  # tolerate a concurrent atomic rewrite
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)
    return None


def _write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def merge_run_into_report(report_dir, rundir):
    """Merge one batch-run folder's current results into the persistent report.

    report_dir holds report.json/report.md (e.g. the repo root); rundir is the
    batch run's own timestamped log folder.

    Self-cleaning, per app: every existing row for an app this batch run runs is
    dropped first, then replaced with only this batch run's jobs for that app. So a
    rerun wipes that app's stale/contaminated/orphan rows (any param combo, any
    leftover queued/building state). Apps the batch run does NOT touch are kept, so
    results still accumulate across batch runs by app (see the Batch run column)."""
    stamp = os.path.basename(os.path.normpath(rundir))
    status = _read_json(os.path.join(rundir, "status.json"))
    if status is None:
        return

    report_path = os.path.join(report_dir, REPORT_JSON)
    report = _read_json(report_path) or {"jobs": {}}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    cur = status.get("jobs", {})
    run_apps = {job["app"] for job in cur.values()}
    report["jobs"] = {jid: e for jid, e in report.get("jobs", {}).items()
                      if e.get("app") not in run_apps}

    for jid, job in cur.items():
        log_abs = os.path.join(rundir, job["log"])
        errors, sc, tot = parse_log(log_abs)
        report["jobs"][jid] = {
            "app": job["app"], "tag": job.get("tag", ""),
            "params": job.get("params", {}),
            "seqLen": job.get("seqLen"), "dModel": job.get("dModel"),
            "n_tiles": job.get("n_tiles"),
            "batch_run": stamp, "log": os.path.relpath(log_abs, report_dir),
            "state": job.get("state", "?"),
            "errors": errors, "simbacore": sc, "total": tot,
            "updated": now,
        }
    report["updated"] = now
    _write_json_atomic(report_path, report)
    # Render the human-readable markdown copy alongside it.
    with open(os.path.join(report_dir, REPORT_MD), "w") as f:
        f.write(render_report(report_dir))


# These get their own columns; don't repeat them in the keyword Params column.
SPECIAL_KEYS = {"seqLen", "dim0", "dModel", "dim1", "nb_tiles", "n_tiles"}


def _fmt_other(params):
    """Render every param that doesn't get its own column, as `k=v`."""
    items = [(k, v) for k, v in (params or {}).items() if k not in SPECIAL_KEYS]
    if not items:
        return "—"
    return ", ".join(f"`{k}={v}`" for k, v in items)


def _fmt_col(v):
    return "—" if v in (None, "") else str(v)


def _fmt_num(s):
    return f"{int(s):,}" if s and s.isdigit() else (s or "—")


def _md_table(headers, aligns, rows):
    """Markdown table, cells padded so the raw source also aligns in a terminal."""
    widths = [max(_dw(h), *(_dw(r[i]) for r in rows)) if rows else _dw(h)
              for i, h in enumerate(headers)]
    rights = [a == "right" for a in aligns]
    out = ["| " + " | ".join(_pad(h, widths[i], rights[i])
                             for i, h in enumerate(headers)) + " |"]
    seps = []
    for i, w in enumerate(widths):
        seps.append(("-" * (w + 1) + ":") if rights[i] else (":" + "-" * (w + 1)))
    out.append("|" + "|".join(seps) + "|")
    for r in rows:
        out.append("| " + " | ".join(_pad(c, widths[i], rights[i])
                                     for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def render_report(report_dir):
    """Render the accumulated report as Markdown from report_dir/report.json."""
    report = _read_json(os.path.join(report_dir, REPORT_JSON))
    if report is None or not report.get("jobs"):
        return f"_No batch-run report yet in {report_dir}._\n"

    jobs = report["jobs"]
    counts = {}
    rows = []
    for e in jobs.values():
        disp = _display_status(e.get("state", "?"), e.get("errors"))
        counts[disp] = counts.get(disp, 0) + 1
        status = f"{EMOJI.get(disp, '')} {disp}".strip()
        rows.append((e["app"], _fmt_col(e.get("seqLen")), _fmt_col(e.get("dModel")),
                     _fmt_col(e.get("n_tiles")), _fmt_other(e.get("params")),
                     e.get("batch_run", "?"), status, e.get("errors") or "—",
                     _fmt_num(e.get("simbacore")), _fmt_num(e.get("total"))))
    rows.sort(key=lambda r: (r[0], r[4]))

    tally = " · ".join(f"{EMOJI.get(k, '')} {v} {k}" for k, v in sorted(counts.items()))
    headers = ["App", "seqLen", "dModel", "n_tiles", "Params", "Batch run",
               "Status", "Errors", "SimbaCore", "Total"]
    aligns = ["left", "right", "right", "right", "left", "left",
              "left", "right", "right", "right"]
    return (
        "# SNAX batch-run report\n\n"
        f"_Updated {report.get('updated', '?')} · {len(jobs)} jobs · {tally}_\n\n"
        + _md_table(headers, aligns, rows) + "\n"
    )


def watch(report_dir, interval):
    while True:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(render_report(report_dir))
        sys.stdout.flush()
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description="Persistent batch-run report")
    ap.add_argument("report_dir", nargs="?", default=".",
                    help="directory holding report.json (default: cwd / repo root)")
    ap.add_argument("--interval", type=float, default=3.0,
                    help="refresh seconds (default 3)")
    ap.add_argument("--once", action="store_true", help="render once and exit")
    ap.add_argument("--merge", metavar="RUNDIR",
                    help="merge a specific batch-run folder into the report, then render")
    args = ap.parse_args()

    if args.merge:
        merge_run_into_report(args.report_dir, args.merge)
        sys.stdout.write(render_report(args.report_dir))
    elif args.once:
        sys.stdout.write(render_report(args.report_dir))
    else:
        watch(args.report_dir, args.interval)


if __name__ == "__main__":
    main()
