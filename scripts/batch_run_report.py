#!/usr/bin/env python3
"""Persistent live report for batch runs (see batch_run.py).

There is ONE report file, accumulated over time across batch runs, at the repo root:
    <root>/report.json   machine-readable, keyed by job id (app__tag)
    <root>/report.md     rendered Markdown table

Each batch run writes its own logs/elfs to <root>/batch_run_out/<timestamp>/, and
merges its results into the single report. A row records which batch run (timestamp)
the stored number came from. The merge is self-cleaning PER JOB (app__tag): a batch
run overwrites only the rows for the exact jobs it ran and keeps every other row, so
configs you removed/commented out persist (flagged stale + sorted down). Rows
accumulate across batch runs; prune report.json by hand to drop dead configs.

Usage (run from the repo root, or pass the dir holding report.json):
    python3 scripts/batch_run_report.py                 # live-watch the report
    python3 scripts/batch_run_report.py --once          # render once
    python3 scripts/batch_run_report.py --merge batch_run_out/<ts>   # import a run folder
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# Markers emitted by the simulator (same ones the regression test parses).
RE_ERRORS = re.compile(r"Finished with exit code\s+(\d+)")
RE_ERRORS_ALT = re.compile(r"Errors:\s+(\d+)")
RE_SIMBACORE = re.compile(r"Simbacore elapsed time:\s+(\d+)\s+cycles")
RE_TOTAL = re.compile(r"Snitch elapsed time:\s+(\d+)\s+cycles")
RE_L1 = re.compile(r"Expected L1 TCDM usage:\s+\d+\s+B\s+\((\d+)\s+KiB\)")
RE_L1_OOM = re.compile(r"L1 TCDM OOM")
# memsim's golden-free AGU audit prints its LOCATED layout-error count to stderr
# (folded into the .memsim.log). This is more informative than the binary exit code
# for the Model Err column: it is the actual number of streamer/AGU layout faults the
# model located (bounds + producer->consumer), not just pass/fail.
RE_MODEL_AGU = re.compile(r"AGU layout audit:\s+(\d+)\s+located error")

REPORT_JSON = "report.json"
REPORT_MD = "report.md"

# Status -> emoji. All chosen to render as double-width glyphs so the raw-text
# table stays aligned in a terminal (markdown viewers ignore the padding).
EMOJI = {
    "PASS": "✅", "ERRORS": "❌", "OOM": "🔴", "RUNNING": "🟢",
    "BUILDING": "🔨", "BUILD_FAIL": "🧱", "TIMEOUT": "🕒", "QUEUED": "🟡",
    "NO_RESULT": "❔",
}
# Appended to the Batch-run cell of any row NOT from the most recent batch run
# (stale, regardless of status). Distinct from TIMEOUT's 🕒 and the commit ⚠️.
STALE_MARK = "⏰"
_WIDE = set(EMOJI.values()) | {"🔴", STALE_MARK}

# Markdown link [text](url): when rendered, only `text` occupies columns, so the
# table's width math must ignore the (often long) url part.
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _dw(s):
    """Display width: emoji occupy two columns but len() counts one; a markdown
    link counts only its visible text (the url is hidden in a rendered view)."""
    s = _LINK_RE.sub(r"\1", s)
    return len(s) + sum(1 for c in s if c in _WIDE)


def _pad(s, width, right=False):
    fill = " " * max(0, width - _dw(s))
    return fill + s if right else s + fill


def parse_log(path):
    """Return (errors, simbacore, total, l1_bytes) strings from a run log, or Nones."""
    if not path or not os.path.exists(path):
        return None, None, None, None
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return None, None, None, None
    m = RE_ERRORS.findall(text) or RE_ERRORS_ALT.findall(text)
    sc = RE_SIMBACORE.findall(text)
    tot = RE_TOTAL.findall(text)
    l1 = RE_L1.findall(text)
    return ((m[-1] if m else None), (sc[-1] if sc else None),
            (tot[-1] if tot else None), (l1[-1] if l1 else None))


def parse_model_agu_errors(path):
    """The memsim AGU audit's located layout-error count from a .memsim.log, or None
    if the line is absent (e.g. a vsim-only log, or --timing-only)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    m = RE_MODEL_AGU.findall(text)
    return m[-1] if m else None


def parse_build_log_l1(path):
    """Read the memory model's predicted L1 peak (KiB) and OOM flag from a build
    log. The model emits this during datagen, so it is the primary source for the
    L1 column -- known at build time, no need to wait for (or even run) the sim,
    and present even when an OOM aborts the build. Returns (l1_kib, oom)."""
    if not path or not os.path.exists(path):
        return None, False
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return None, False
    l1 = RE_L1.findall(text)
    return (l1[-1] if l1 else None), bool(RE_L1_OOM.search(text))


def _display_status(state, errors, oom=False):
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
        if errors == "0":
            return "PASS"
        # Nonzero error count: an OOM-predicted run gets the clearer OOM label;
        # otherwise ERRORS (a small count may be quantization noise, not a true fail).
        return "OOM" if oom else "ERRORS"
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

    Self-cleaning, per JOB (app__tag): this run overwrites only the rows for the
    exact jobs it ran; every other row is kept. So configs you removed or commented
    out keep their last result instead of being wiped when a sibling config of the
    same app reruns -- they're just flagged stale (⏰) and sorted to the bottom.
    Rows accumulate across runs; prune report.json by hand to drop dead configs."""
    stamp = os.path.basename(os.path.normpath(rundir))
    status = _read_json(os.path.join(rundir, "status.json"))
    if status is None:
        return
    commit = status.get("commit")

    report_path = os.path.join(report_dir, REPORT_JSON)
    report = _read_json(report_path) or {"jobs": {}}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    cur = status.get("jobs", {})
    # Self-cleaning is PER JOB, not per app: this run overwrites only the exact jobs
    # (app__tag) it ran; every other row is kept. So a config you removed or commented
    # out persists with its last result (flagged stale + sorted to the bottom) instead
    # of being wiped just because a sibling config of the same app reran.
    report.setdefault("jobs", {})

    for jid, job in cur.items():
        log_abs = os.path.join(rundir, job["log"])
        errors, sc, tot, sim_l1 = parse_log(log_abs)
        # The cycle-accurate memsim model runs alongside the vsim into its own log;
        # scrape the same markers for the Model error/SimbaCore/Total columns.
        memsim_log = os.path.join(rundir, os.path.splitext(job["log"])[0] + ".memsim.log")
        m_errors, m_sc, m_tot, _ = parse_log(memsim_log)
        # Prefer the AGU audit's LOCATED layout-error count over the binary exit code
        # for Model Err -- it says how many faults the model found, not just 0/1. Guard:
        # if the audit located 0 but the run still failed (a non-AGU model check, exit
        # nonzero), keep the failure visible rather than masking it with a clean 0.
        agu = parse_model_agu_errors(memsim_log)
        if agu is not None and not (agu == "0" and m_errors not in (None, "0")):
            m_errors = agu
        # L1 TCDM peak is a STATIC prediction the memory model emits during the build
        # (datagen) -- read it straight from the build log rather than waiting for the
        # sim to re-print the baked constant (whose format also varies per app, and
        # which never appears at all when an OOM aborts the build). The OOM flag comes
        # from the same place. Fall back to the sim log only if the app has no memory model.
        build_log = os.path.join(rundir, os.path.splitext(job["log"])[0] + ".build.log")
        l1, oom = parse_build_log_l1(build_log)
        if l1 is None:
            l1 = sim_l1
        report["jobs"][jid] = {
            "app": job["app"], "tag": job.get("tag", ""),
            "params": job.get("params", {}),
            "seqLen": job.get("seqLen"), "dModel": job.get("dModel"),
            "n_tiles": job.get("n_tiles"),
            "batch_run": stamp, "commit": commit,
            "log": os.path.relpath(log_abs, report_dir),
            "state": job.get("state", "?"),
            "errors": errors, "simbacore": sc, "total": tot,
            "model_errors": m_errors, "model_simbacore": m_sc, "model_total": m_tot,
            "l1_kib": l1, "l1_oom": oom,
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
    return ", ".join(f"{k}={v}" for k, v in items)


def _fmt_col(v):
    return "—" if v in (None, "") else str(v)


def _fmt_num(s):
    return f"{int(s):,}" if s and s.isdigit() else (s or "—")


def _fmt_l1(s, oom=False):
    """L1 TCDM peak (KiB, as the app printed it) -> a `<KiB> KiB` cell, or em-dash.
    Appends an OOM marker when the predicted peak exceeds the TCDM budget."""
    if not (s and str(s).isdigit()):
        return "—"
    return f"{int(s):,} KiB" + (" 🔴OOM" if oom else "")


def _log_link(report_dir, log_rel):
    """Clickable link to a job's most useful log: the sim log if it exists, else
    the build log (e.g. a build failure never produced a sim log). Paths are
    relative to report_dir so the link resolves from report.md living there."""
    if not log_rel:
        return "—"
    build_rel = os.path.splitext(log_rel)[0] + ".build.log"
    if os.path.exists(os.path.join(report_dir, log_rel)):
        return f"[log]({log_rel})"
    if os.path.exists(os.path.join(report_dir, build_rel)):
        return f"[build]({build_rel})"
    return "—"


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


def current_commit(report_dir):
    """Short HEAD of the repo the report lives in, to flag rows built off old commits."""
    try:
        out = subprocess.check_output(["git", "-C", report_dir, "rev-parse", "--short", "HEAD"])
        return out.decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def render_report(report_dir):
    """Render the accumulated report as Markdown from report_dir/report.json."""
    report = _read_json(os.path.join(report_dir, REPORT_JSON))
    if report is None or not report.get("jobs"):
        return f"_No batch-run report yet in {report_dir}._\n"

    head = current_commit(report_dir)
    jobs = report["jobs"]
    # The most recent batch_run stamp = the run currently in progress (or, when
    # viewing standalone, the last one). Rows from earlier runs are accumulated
    # leftovers, not what we ran now -- mark their PASS distinctly.
    stamps = [e.get("batch_run") for e in jobs.values() if e.get("batch_run")]
    latest_run = max(stamps) if stamps else None
    counts = {}
    rows = []
    for e in jobs.values():
        disp = _display_status(e.get("state", "?"), e.get("errors"), e.get("l1_oom"))
        counts[disp] = counts.get(disp, 0) + 1
        status = f"{EMOJI.get(disp, '')} {disp}".strip()
        commit = e.get("commit")
        # Flag rows whose run predates the current HEAD -- their numbers may be stale.
        if not commit:
            commit_cell = "—"
        elif head and commit != head:
            commit_cell = f"{commit} ⚠️"
        else:
            commit_cell = commit
        # Stale = not from the most recent batch run (any status): clock the batch-run
        # cell and sort these to the bottom.
        stale = bool(latest_run) and e.get("batch_run") != latest_run
        batch_cell = e.get("batch_run", "?")
        if stale:
            batch_cell = f"{batch_cell} {STALE_MARK}"
        row = (e["app"], _fmt_col(e.get("seqLen")), _fmt_col(e.get("dModel")),
               _fmt_col(e.get("n_tiles")), _fmt_other(e.get("params")),
               batch_cell, commit_cell, status, e.get("errors") or "—",
               _fmt_num(e.get("simbacore")), _fmt_num(e.get("total")),
               e.get("model_errors") or "—",
               _fmt_num(e.get("model_simbacore")), _fmt_num(e.get("model_total")),
               _fmt_l1(e.get("l1_kib"), e.get("l1_oom")),
               _log_link(report_dir, e.get("log")))
        rows.append((stale, row))
    # Fresh rows first (stale at the bottom); within each group by app then params.
    rows.sort(key=lambda sr: (sr[0], sr[1][0], sr[1][4]))
    rows = [r for _, r in rows]

    tally = " · ".join(f"{EMOJI.get(k, '')} {v} {k}" for k, v in sorted(counts.items()))
    headers = ["App", "seqLen", "dModel", "n_tiles", "Params", "Batch run",
               "Commit", "Status", "Errors", "SimbaCore", "Total",
               "Model Err", "Model SimbaCore", "Model Total", "L1 TCDM", "Log"]
    aligns = ["left", "right", "right", "right", "left", "left",
              "left", "left", "right", "right", "right",
              "right", "right", "right", "right", "left"]
    head_note = f" · HEAD `{head}`" if head else ""
    return (
        "# SNAX batch-run report\n\n"
        f"_Updated {report.get('updated', '?')} · {len(jobs)} jobs{head_note} · {tally}_\n\n"
        "_⚠️ = run predates current HEAD (numbers may be stale)._\n\n"
        "_⏰ next to the batch run = result from an earlier batch run (stale; sorted to the bottom)._\n\n"
        "_❌ ERRORS = nonzero mismatch count (small values may be quantization noise, not a true fail) · 🔴 OOM = exceeded L1 TCDM budget._\n\n"
        "_Errors/SimbaCore/Total = RTL vsim · Model Err/SimbaCore/Total = cycle-accurate memsim model (memsim) on the same .elf._\n\n"
        "_Model Err = count of AGU/layout faults the model LOCATED (bounds + producer→consumer); 0 = layout clean._\n\n"
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
