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
# vsim's own end-of-run summary. Its count is simulator faults ($error/$fatal/assert),
# NOT the app's output-compare count -- any nonzero value is a real crash, not
# quantization noise, so it is never eligible for the FP8 tolerance below.
RE_VSIM_ERRORS = re.compile(r"Errors:\s+(\d+)")
RE_SIMBACORE = re.compile(r"Simbacore elapsed time:\s+(\d+)\s+cycles")
RE_TOTAL = re.compile(r"Snitch elapsed time:\s+(\d+)\s+cycles")
RE_L1 = re.compile(r"Expected L1 TCDM usage:\s+\d+\s+B\s+\((\d+)\s+KiB\)")
RE_L1_OOM = re.compile(r"L1 TCDM OOM")
# memsim's golden-free AGU audit prints its LOCATED layout-error count to stderr
# (folded into the .memsim.log). This is more informative than the binary exit code
# for the Model Err column: it is the actual number of streamer/AGU layout faults the
# model located (bounds + producer->consumer), not just pass/fail.
RE_MODEL_AGU = re.compile(r"AGU layout audit:\s+(\d+)\s+located error")
# memsim's safe-to-start co-sim prints the stale-read count at the app's actual start_cnts:
# a config that releases a consumer before its producer has committed reads uncommitted data
# (exactly the gross iscore_out/SUC-y vsim failures). This is the model's prediction that a
# config produces WRONG OUTPUT, and it belongs in Model Err alongside the AGU faults.
RE_MODEL_S2S = re.compile(r"stale_z=(\d+)\s+stale_y=(\d+)")

REPORT_JSON = "report.json"
REPORT_MD = "report.md"

# Pass if errors < ERROR_THRESHOLD
ERROR_THRESHOLD = 5


EMOJI = {
    "PASS": "✅",
    "ERRORS": "❌",
    "OOM": "🔴",
    "RUNNING": "🟢",
    "BUILDING": "🔨",
    "BUILD_FAIL": "🧱",
    "TIMEOUT": "🕒",
    "QUEUED": "🟡",
    "NO_RESULT": "❔",
}

STALE_MARK = "⏰"
CURRENT_MARK = "✨"
CACHED_MARK = "💾"
_WIDE = set(EMOJI.values()) | {"🔴", STALE_MARK, CURRENT_MARK, CACHED_MARK}

# Markdown link [text](url)
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
    """Return (errors, simbacore, total, l1_bytes, crashed) from a run log, or Nones.

    `errors` is the app's own output-compare count from the "Finished with exit code N"
    completion marker -- the only number the FP8 quantization tolerance applies to. A run
    that never printed that marker did not complete; `crashed` is True when the log instead
    ends on a simulator fault (vsim "Errors: N>0", e.g. a fatal RTL assertion), so it is
    failed outright rather than judged against the tolerance."""
    if not path or not os.path.exists(path):
        return None, None, None, None, False
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return None, None, None, None, False
    comp = RE_ERRORS.findall(text)
    sc = RE_SIMBACORE.findall(text)
    tot = RE_TOTAL.findall(text)
    l1 = RE_L1.findall(text)
    if comp:
        errors, crashed = comp[-1], False
    else:
        vsim = RE_VSIM_ERRORS.findall(text)
        errors = vsim[-1] if vsim else None
        crashed = bool(vsim) and int(vsim[-1]) > 0
    return (errors, (sc[-1] if sc else None), (tot[-1] if tot else None), (l1[-1] if l1 else None), crashed)


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


def parse_model_s2s_stale(path):
    """Total stale reads (z + y) the safe-to-start co-sim located at the app's own
    start_cnts, or None if the line is absent (not a P2 osCore->SUC->isCore app).
    Nonzero = the model predicts this config releases a consumer too early and reads
    uncommitted data -> wrong output (the gross vsim iscore_out/SUC-y failures)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    m = RE_MODEL_S2S.findall(text)
    if not m:
        return None
    z, y = m[-1]
    return int(z) + int(y)


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


def _display_status(state, errors, oom=False, crashed=False):
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
        if crashed:
            return "ERRORS"
        if errors is None:
            return "NO_RESULT"
        if errors.isdigit() and int(errors) < ERROR_THRESHOLD:
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
        cached = job.get("cached", False)
        prev = report["jobs"].get(jid, {})  # existing row (carries the reused result)
        if cached:
            # vsim was reused (no build/vsim), but the fast memsim model re-runs every batch on the
            # cached elf -> keep the stored vsim columns but refresh the Model columns from the FRESH
            # memsim log when present (else keep the stored row verbatim). Re-stamped -> renders 💾.
            row = {
                **prev,
                # `name` is display-only (popped before the app__tag jid is computed),
                # so renaming a config reuses the cached vsim under the same jid. Refresh
                # it from the fresh job so a rename shows up in the report.
                "name": job.get("name"),
                "model": job.get("model"),
                "batch_run": stamp,
                "cached": True,
                "state": prev.get("state", job.get("state", "done")),
                "updated": now,
            }
            memsim_log = os.path.join(rundir, os.path.splitext(job["log"])[0] + ".memsim.log")
            if os.path.exists(memsim_log):
                m_errors, m_sc, m_tot, _, _ = parse_log(memsim_log)
                # Only overwrite the stored Model columns when the fresh memsim actually produced a
                # SimbaCore number; a timeout / unparseable / config-mismatch run must not blank the row.
                if m_sc is not None:
                    agu = parse_model_agu_errors(memsim_log)
                    stale = parse_model_s2s_stale(memsim_log)
                    if agu is not None or stale is not None:
                        m_errors = str(int(agu or 0) + int(stale or 0))
                    row.update(model_errors=m_errors, model_simbacore=m_sc, model_total=m_tot)
                    plot_abs = os.path.join(rundir, os.path.splitext(job["log"])[0] + ".timeline.png")
                    if os.path.exists(plot_abs):
                        row["timeline"] = os.path.relpath(plot_abs, report_dir)
            report["jobs"][jid] = row
            continue
        log_abs = os.path.join(rundir, job["log"])
        errors, sc, tot, sim_l1, crashed = parse_log(log_abs)
        # memsim runs alongside the vsim into its own log. Scrape the same markers
        memsim_log = os.path.join(rundir, os.path.splitext(job["log"])[0] + ".memsim.log")
        m_errors, m_sc, m_tot, _, _ = parse_log(memsim_log)
        # Model Err = the faults the model LOCATED, not its binary exit code (which a
        # tiled-golden false positive can flip). Two located sources, summed:
        #   - the golden-free AGU audit (bounds + producer->consumer + writer no-alias), and
        #   - the safe-to-start co-sim's stale reads at the app's own start_cnts (a config
        #     that releases a consumer too early -> reads uncommitted data -> wrong output).
        # When either line is present the model ran, so the sum is authoritative; only fall
        # back to the exit code for logs with no audit at all (vsim-only / --timing-only).
        agu = parse_model_agu_errors(memsim_log)
        stale = parse_model_s2s_stale(memsim_log)
        if agu is not None or stale is not None:
            m_errors = str(int(agu or 0) + int(stale or 0))
        # L1 TCDM peak is a STATIC prediction the memory model emits during the build
        # (datagen) -- read it straight from the build log rather than waiting for the
        # sim to re-print the baked constant (whose format also varies per app, and
        # which never appears at all when an OOM aborts the build). The OOM flag comes
        # from the same place. Fall back to the sim log only if the app has no memory model.
        build_log = os.path.join(rundir, os.path.splitext(job["log"])[0] + ".build.log")
        l1, oom = parse_build_log_l1(build_log)
        if l1 is None:
            l1 = sim_l1
        # memsim's per-engine activity + TCDM-bandwidth timeline plot, rendered alongside
        # the memsim log (sibling of the sim log). Absent for older runs / when memsim or
        # matplotlib was unavailable -> stored as None (cell shows em-dash).
        plot_abs = os.path.join(rundir, os.path.splitext(job["log"])[0] + ".timeline.png")
        log_rel = os.path.relpath(log_abs, report_dir)
        report["jobs"][jid] = {
            "app": job["app"],
            "tag": job.get("tag", ""),
            "name": job.get("name"),
            "model": job.get("model"),
            "params": job.get("params", {}),
            "seqLen": job.get("seqLen"),
            "dModel": job.get("dModel"),
            "n_tiles": job.get("n_tiles"),
            "batch_run": stamp,
            "commit": commit,
            "log": log_rel,
            "timeline": os.path.relpath(plot_abs, report_dir) if os.path.exists(plot_abs) else None,
            "state": job.get("state", "?"),
            "cached": cached,
            "errors": errors,
            "crashed": crashed,
            "simbacore": sc,
            "total": tot,
            "model_errors": m_errors,
            "model_simbacore": m_sc,
            "model_total": m_tot,
            "l1_kib": l1,
            "l1_oom": oom,
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
    return ", ".join(f"{k.replace('safe_to_start_', 's2s_')}={v}" for k, v in items)


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


def _plot_link(report_dir, plot_rel):
    """Clickable link to a job's memsim timeline figure, or em-dash if absent. Path is
    relative to report_dir so the link resolves from report.md living there."""
    if plot_rel and os.path.exists(os.path.join(report_dir, plot_rel)):
        return f"[plot]({plot_rel})"
    return "—"


def _md_table(headers, aligns, rows):
    """Markdown table, cells padded so the raw source also aligns in a terminal."""
    widths = [max(_dw(h), *(_dw(r[i]) for r in rows)) if rows else _dw(h) for i, h in enumerate(headers)]
    rights = [a == "right" for a in aligns]
    out = ["| " + " | ".join(_pad(h, widths[i], rights[i]) for i, h in enumerate(headers)) + " |"]
    seps = []
    for i, w in enumerate(widths):
        seps.append(("-" * (w + 1) + ":") if rights[i] else (":" + "-" * (w + 1)))
    out.append("|" + "|".join(seps) + "|")
    for r in rows:
        out.append("| " + " | ".join(_pad(c, widths[i], rights[i]) for i, c in enumerate(r)) + " |")
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
        disp = _display_status(e.get("state", "?"), e.get("errors"), e.get("l1_oom"), e.get("crashed"))
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
        # Stale = not from the most recent batch run (any status): sort these to the
        # bottom. The cell is just a marker, not the long timestamp (it's in report.json).
        stale = bool(latest_run) and e.get("batch_run") != latest_run
        if stale:
            batch_cell = STALE_MARK
        elif e.get("cached"):
            batch_cell = CACHED_MARK
        else:
            batch_cell = CURRENT_MARK
        row = (
            _fmt_col(e.get("name")),
            _fmt_col(e.get("model")),
            e["app"],
            _fmt_col(e.get("seqLen")),
            _fmt_col(e.get("dModel")),
            _fmt_col(e.get("n_tiles")),
            _fmt_other(e.get("params")),
            batch_cell,
            commit_cell,
            status,
            e.get("errors") or "—",
            _fmt_num(e.get("simbacore")),
            _fmt_num(e.get("total")),
            e.get("model_errors") or "—",
            _fmt_num(e.get("model_simbacore")),
            _fmt_num(e.get("model_total")),
            _fmt_l1(e.get("l1_kib"), e.get("l1_oom")),
            _log_link(report_dir, e.get("log")),
            _plot_link(report_dir, e.get("timeline")),
        )
        rows.append((stale, row))
    # Fresh rows first (stale at the bottom). Sort by user-define name, then app name
    rows.sort(key=lambda sr: (sr[0], sr[1][0], sr[1][2]))
    rows = [r for _, r in rows]

    tally = " · ".join(f"{EMOJI.get(k, '')} {v} {k}" for k, v in sorted(counts.items()))
    headers = [
        "Name",
        "Model",
        "App",
        "seqLen",
        "dModel",
        "n_tiles",
        "Params",
        "Run",
        "Commit",
        "Status",
        "Errors",
        "SimbaCore",
        "Total",
        "Model Err",
        "Model SimbaCore",
        "Model Total",
        "L1 TCDM",
        "Log",
        "Plot",
    ]
    aligns = [
        "left",
        "left",
        "left",
        "right",
        "right",
        "right",
        "left",
        "left",
        "left",
        "left",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "left",
        "left",
    ]
    head_note = f" · HEAD `{head}`" if head else ""
    return (
        "# SNAX batch-run report\n\n"
        f"_Updated {report.get('updated', '?')} · {len(jobs)} jobs{head_note} · {tally}_\n\n"
        + _md_table(headers, aligns, rows)
        + "\n"
    )


def watch(report_dir, interval):
    while True:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(render_report(report_dir))
        sys.stdout.flush()
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description="Persistent batch-run report")
    ap.add_argument(
        "report_dir", nargs="?", default=".", help="directory holding report.json (default: cwd / repo root)"
    )
    ap.add_argument("--interval", type=float, default=3.0, help="refresh seconds (default 3)")
    ap.add_argument("--once", action="store_true", help="render once and exit")
    ap.add_argument("--merge", metavar="RUNDIR", help="merge a specific batch-run folder into the report, then render")
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
