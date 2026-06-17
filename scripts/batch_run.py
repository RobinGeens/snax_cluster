#!/usr/bin/env python3
"""Run a whole batch of simulations from one config file.

Reads an hjson config listing apps and, per app, a list of param-override sets.
For each (app, override-set) it: generates a temp params file (the tracked
params_in.hjson as a read-only base + the config overrides), builds the app's
.elf (in the podman container) with the build pointed at that temp file via the
PARAMS_IN env var, stages the .elf to a unique path, then launches the vsim. The
tracked params_in.hjson is never modified. Each run gets its own log file.

Scheduling:
  * A build's slow part is the chisel-ssm/sbt datagen, and only its *first*
    occurrence for a given data shape pays it -- the result is cached under
    .datagen_cache/ and every later build of that shape just untars it. sbt is
    memory-heavy and cannot run two at a time. So builds split into two lanes,
    classified up front by replicating the makefile's datagen-cache check:
      - Datagen lane: ONE serial worker for jobs whose data is NOT yet cached
        (each runs at most one sbt). A slow datagen here no longer blocks others.
      - Fast-build pool: `build_parallel` workers for jobs whose data IS cached
        (guaranteed cache hit -> no sbt -> just compile). These run concurrently.
    The two lanes run at the same time, so a 2-hour datagen for one shape no
    longer stalls dozens of already-cached compiles.
  * Builds of the SAME app are still serialized (shared per-app build dir +
    generated/data dir); different apps build in parallel. So same-app cached
    builds wait behind that app's slow datagen, but every other app proceeds.
  * Sims run concurrently, bounded by `max_parallel` (the OOM knob).
  * A build does not wait for its sim to finish before the next build starts.

Results are merged into ONE persistent report at the repo root (<root>/report.md
+ report.json). The merge is self-cleaning per JOB (app__tag): rerunning a config
overwrites only its own row and keeps every other, so configs you removed/commented
out persist (flagged stale + sorted to the bottom). A Batch run column records which
timestamped batch run each number came from. View it
separately with:
    python3 scripts/batch_run_report.py            # run from the repo root

Usage (run from anywhere in the repo):
    python3 scripts/batch_run.py                       # uses <root>/batch_run_config.hjson
    python3 scripts/batch_run.py path/to/other.hjson
"""
import argparse
import fcntl
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone

import hjson

import batch_run_report

CONTAINER = "ghcr.io/kuleuven-micas/snax:main"
VSIM_BIN = "bin/snitch_cluster.vsim"
# Cycle-accurate memory-simulator model: a host-side drop-in for the .vsim (no
# podman, no license, orders of magnitude faster). Run per job alongside the
# vsim; its Simbacore/Snitch/error lines are scraped into the report's Model cols.
MEMSIM_BIN = "bin/snitch_cluster.memsim"


def repo_root():
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    return out.decode().strip()


def git_commit(root):
    """Short HEAD hash of the repo a run is built from, for staleness tracking."""
    try:
        out = subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"])
        return out.decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def slug(value):
    return str(value).replace("/", "-").replace(" ", "")


def make_tag(overrides):
    if not overrides:
        return "default"
    return "_".join(f"{k}{slug(v)}" for k, v in sorted(overrides.items()))


def _pick(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


class BatchRun:
    def __init__(self, config_path=None, no_redo=False, skip_lock=False):
        self.cli_no_redo = no_redo
        self.root = repo_root()
        self.cluster = os.path.join(self.root, "target", "snitch_cluster")
        # Only ONE batch run may run at a time: concurrent batch runs share each
        # app's build dir, the chisel-ssm/sbt datagen cache and the root
        # report.json -- running two corrupts builds and results. Take an
        # exclusive lock that the OS releases automatically if we die. --remodel
        # skips it: it neither builds, vsims, nor touches params_in, and the report
        # is rewritten atomically, so it is safe alongside a live run.
        self._lock_fd = None
        if not skip_lock:
            self._lock_fd = open(os.path.join(self.root, ".batch_run.lock"), "w")
            try:
                fcntl.lockf(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                held = open(os.path.join(self.root, ".batch_run.lock")).read().strip()
                sys.exit(
                    f"Another batch run is already running (PID {held or '?'}). "
                    f"Refusing to start a second one -- it would corrupt the "
                    f"shared params_in.hjson and report. Lock: {self.root}/.batch_run.lock"
                )
            self._lock_fd.seek(0)
            self._lock_fd.truncate()
            self._lock_fd.write(str(os.getpid()))
            self._lock_fd.flush()

        if config_path is None:
            config_path = os.path.join(self.root, "batch_run_config.hjson")
        self.config_path = os.path.abspath(config_path)
        with open(self.config_path) as f:
            self.cfg = hjson.load(f)

        self.max_parallel = int(self.cfg.get("max_parallel", 2))
        # Number of concurrent fast (cache-hit) compiles. The serial datagen lane
        # runs alongside these, so total concurrent builds peak at build_parallel+1.
        self.build_parallel = int(self.cfg.get("build_parallel", 4))
        # Per-run wall-clock cap in seconds; 0 disables (default: none).
        self.timeout = int(self.cfg.get("timeout", 0))
        # Build (podman make) wall-clock cap; 0 disables
        self.build_timeout = int(self.cfg.get("build_timeout", 0))
        # Global default for `force`; an individual param-set may override it with its
        # own `force` key. `force: false` (or --no-redo) => fully skip a job whose last
        # run already produced a result: no build, no vsim, no memsim -- the report just
        # keeps its last stored row verbatim. build_fail / timeout / no_result always rerun.
        self.force = bool(self.cfg.get("force", True))
        self.cached_jobs = []

        # The single, persistent report lives at the repo root and accumulates
        # across batch runs; each run's logs/elfs go in their own timestamped dir.
        self.report_dir = self.root
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.rundir = os.path.join(self.root, "batch_run_out", ts)
        os.makedirs(self.rundir, exist_ok=True)

        self.sim_sem = threading.Semaphore(self.max_parallel)
        self.status_lock = threading.Lock()
        self.sim_threads = []
        self.sim_threads_lock = threading.Lock()

        # Every child (podman build, vsim) is launched in its own process group
        # and tracked here so the whole tree can be torn down on cancel.
        self.procs = {}  # pid -> Popen
        self.procs_lock = threading.Lock()
        self.aborted = threading.Event()
        # Names of live build containers. Killing the `podman run` client does NOT
        # stop the container (conmon keeps it alive, reparented to init), so we
        # name each one and `podman rm -f` it on teardown -- the only reliable way
        # to kill an orphaned build. Guarded by procs_lock.
        self.build_containers = set()
        self._build_seq = 0
        # Per-app build lock: all builds of one app (datagen lane + fast pool)
        # share its generated/data + build dir, so only one may `make` at a time.
        # Different apps build concurrently. _app_locks_guard guards lazy creation.
        self._app_locks = defaultdict(threading.Lock)
        self._app_locks_guard = threading.Lock()

        self.jobs = self._build_job_list()  # ordered list per lane
        self._mark_cached()  # tag jobs whose vsim should be reused (still built + memsim'd)
        self.status = {
            "rundir": self.rundir,
            "config_path": self.config_path,
            "started": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "max_parallel": self.max_parallel,
            "commit": git_commit(self.root),
            "complete": False,
            "jobs": {},
        }
        for lane in self.jobs:
            for job in lane:
                self.status["jobs"][job["id"]] = {
                    "app": job["app"],
                    "tag": job["tag"],
                    "name": job["name"],
                    "overrides": dict(job["overrides"]),
                    "params": job["all_params"],
                    "seqLen": job["seqLen"],
                    "dModel": job["dModel"],
                    "n_tiles": job["n_tiles"],
                    "log": job["id"] + ".log",
                    # Cached = fully skipped this run (no build/vsim/memsim); the merge
                    # keeps the stored row verbatim. Start it `done` so it never renders
                    # as queued -- nothing will run for it.
                    "cached": job["cached"],
                    "state": "done" if job["cached"] else "queued",
                    "rc": None,
                }
        self._write_status()

    def _build_job_list(self):
        """Return a list of lanes; each lane is an ordered list of job dicts."""
        lanes = []
        seen_ids = set()
        missing_errors = []  # strict-validation failures, reported all at once
        for entry in self.cfg["runs"]:
            app = entry["app"]
            # Base params = the app's tracked params_in.hjson. Every key here must
            # be given explicitly by each config param-set: a config must be a
            # complete spec, never silently inheriting a value (e.g. dInner) from
            # params_in.hjson, since a stale default produces wrong/invalid data.
            base = {}
            bpath = self.params_path(app)
            if os.path.exists(bpath):
                with open(bpath) as f:
                    base = hjson.load(f)
            param_sets = entry.get("params") or [{}]
            lane = []
            for overrides in param_sets:
                overrides = dict(overrides)
                # `force` is a scheduling directive, not a workload param: pull it out
                # before validation/datagen so it never reaches params/data.h. None =
                # inherit the global `force`.
                force = overrides.pop("force", None)
                # `name` is an optional user-defined label for the report's Name column;
                # like `force` it is display-only, so pull it out before validation/datagen.
                name = overrides.pop("name", None)
                missing = [k for k in base if k not in overrides]
                if missing:
                    missing_errors.append(
                        f"  {app}: param-set {dict(overrides) or '{}'} is missing "
                        f"{missing} (present in {app}/data/params_in.hjson)"
                    )
                eff = {**base, **overrides}
                tag = make_tag(overrides)
                jid = f"{app}__{tag}"
                n = 2
                while jid in seen_ids:  # disambiguate duplicate override sets
                    jid = f"{app}__{tag}__{n}"
                    n += 1
                seen_ids.add(jid)
                lane.append(
                    {
                        "id": jid,
                        "app": app,
                        "tag": tag,
                        "name": name,  # optional user-defined label for the report
                        "force": force,  # None => inherit global; else per-set override
                        "overrides": overrides,
                        "all_params": dict(eff),
                        "seqLen": _pick(eff, "seqLen", "dim0"),
                        "dModel": _pick(eff, "dModel", "dim1"),
                        "n_tiles": _pick(eff, "nb_tiles", "n_tiles"),
                    }
                )
            lanes.append(lane)
        if missing_errors:
            sys.exit(
                "Config error: every param-set must specify all params present in "
                "the app's params_in.hjson (no silent inheritance).\n" + "\n".join(missing_errors)
            )
        return lanes

    def _job_force(self, job):
        """Effective force for one job: --no-redo on the CLI forces it off for every
        job; otherwise the param-set's own `force` if it set one, else the global
        config `force`."""
        if self.cli_no_redo:
            return False
        return self.force if job["force"] is None else bool(job["force"])

    def _mark_cached(self):
        """Tag jobs that should be fully skipped rather than rerun (sets job["cached"]).

        A job is cached iff its effective force is false AND its last recorded run in the
        persistent report.json reached state `done` with a real error count (PASS or
        ERRORS/OOM). force=true jobs (the default) are never cached. build_fail, timeout
        and no_result (state `done` but no error line) always rerun.

        Cached jobs are skipped completely: no .elf build, no vsim, no memsim model. The
        report merge keeps their last stored row verbatim (re-stamped as reused on this run)."""
        report = {}
        report_path = os.path.join(self.report_dir, "report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path) as f:
                    report = json.load(f).get("jobs", {})
            except (OSError, json.JSONDecodeError):
                report = {}

        kept, cached = [], []
        for lane in self.jobs:
            for job in lane:
                prev = report.get(job["id"], {})
                produced_result = prev.get("state") == "done" and prev.get("errors") is not None
                job["cached"] = (not self._job_force(job)) and produced_result
                (cached if job["cached"] else kept).append(job["id"])
        self.cached_jobs = cached
        # Stashed for the end-of-run summary (the live render clears the screen, so a
        # print here would be wiped immediately).
        self.no_redo_summary = (kept, cached)

    # --- status helpers ---
    def _write_status(self):
        tmp = os.path.join(self.rundir, "status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.status, f, indent=2)
        os.replace(tmp, os.path.join(self.rundir, "status.json"))

    def set_state(self, jid, state, rc=None):
        with self.status_lock:
            self.status["jobs"][jid]["state"] = state
            if rc is not None:
                self.status["jobs"][jid]["rc"] = rc
            self._write_status()

    # --- paths ---
    def params_path(self, app):
        return os.path.join(self.cluster, "sw", "apps", app, "data", "params_in.hjson")

    def built_elf(self, app):
        return os.path.join(self.cluster, "sw", "apps", app, "build", app + ".elf")

    # --- build / run primitives ---
    def write_temp_params(self, job):
        """Write the job's full params to a temp file in the rundir and return
        its path (or None for a params-less app like nop). The tracked
        params_in.hjson is never modified -- the build reads this file instead,
        via the PARAMS_IN env var. The temp file lives under the repo root so it
        is visible at the same path inside the podman container."""
        if not os.path.exists(self.params_path(job["app"])):
            if job["overrides"]:
                raise FileNotFoundError(
                    f"{job['app']} has no params_in.hjson but overrides were " f"given: {job['overrides']}"
                )
            return None  # params-less app (e.g. nop) with empty overrides
        path = os.path.join(self.rundir, job["id"] + ".params.hjson")
        with open(path, "w") as f:
            hjson.dump(job["all_params"], f)
        return path

    # --- child process management (every child runs in its own process group
    #     so cancelling the orchestrator tears down the whole bash+vsim tree) ---
    def _spawn(self, cmd, stdout):
        p = subprocess.Popen(cmd, cwd=self.cluster, stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
        with self.procs_lock:
            self.procs[p.pid] = p
        return p

    def _reap(self, p):
        with self.procs_lock:
            self.procs.pop(p.pid, None)

    @staticmethod
    def _killpg(p, sig):
        try:
            os.killpg(os.getpgid(p.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _podman_rm(name):
        """Force-remove a build container by name. The container outlives its
        `podman run` client (conmon detaches it), so killing the client's process
        group is not enough -- this is what actually stops an orphaned build."""
        try:
            subprocess.run(
                ["podman", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            pass

    def terminate_all(self):
        """Kill every tracked child's process group (SIGTERM, then SIGKILL)."""
        self.aborted.set()
        with self.procs_lock:
            procs = list(self.procs.values())
        for p in procs:
            self._killpg(p, signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._killpg(p, signal.SIGKILL)
        # Killing the client leaves the container running -- remove it explicitly.
        with self.procs_lock:
            names = list(self.build_containers)
        for name in names:
            self._podman_rm(name)

    # Scheduling keys that common-datagen.mk filters OUT of the datagen cache key
    # (they affect data.h packing / safe-to-start, not the sbt golden data shape).
    _CACHE_KEY_DROP = ("nb_tiles", "nb_l_tiles", "nb_slots", "safe_to_start", "force")

    def _cache_args(self, job):
        """Replicate common-datagen.mk's CACHE_ARGS: the job's effective params as
        `key=value`, in params-file order, minus the scheduling keys. Used only to
        decide which lane a job goes in -- the build's own make re-checks the cache
        for correctness, so a misclassification only ever costs scheduling, never data."""
        parts = [
            f"{k}={v}"
            for k, v in job["all_params"].items()
            if not any(k.startswith(p) for p in self._CACHE_KEY_DROP)
        ]
        return " ".join(parts)

    def _datagen_ready(self, job):
        """True iff this job's sbt golden data is already cached -- i.e. its `make`
        will hit .datagen_cache and run NO sbt. Mirrors the makefile check exactly
        (md5(CACHE_ARGS) names the tar, and the stored .args must match), so it can
        never report a hit where make would miss => the fast pool never spawns sbt."""
        if not os.path.exists(self.params_path(job["app"])):
            return True  # params-less app (e.g. nop): no datagen at all
        cargs = self._cache_args(job)
        cache_dir = os.path.join(self.cluster, ".datagen_cache", job["app"])
        # The makefile names the tar md5("$CACHE_ARGS\n") (echo adds a newline) but
        # writes the .args body with printf (no newline) -- mirror both exactly.
        key = hashlib.md5((cargs + "\n").encode()).hexdigest()
        tar = os.path.join(cache_dir, key + ".tar")
        args_file = os.path.join(cache_dir, key + ".args")
        if not os.path.exists(tar):
            return False
        try:
            with open(args_file) as f:
                return f.read() == cargs
        except OSError:
            return False

    def _app_lock(self, app):
        with self._app_locks_guard:
            return self._app_locks[app]

    def build(self, app, params_path, blog):
        # PARAMS_IN points both make (WORKLOAD_PARAMS) and datagen.py at the
        # generated temp params file instead of the tracked params_in.hjson.
        env_args = ["-e", f"PARAMS_IN={params_path}"] if params_path else []
        # `make clean` first: the repo lives on NFS and the build runs in podman,
        # so make's mtime-based incremental rebuild is unreliable across jobs --
        # a freshly written temp params file can appear OLDER than an existing
        # data.h, so make skips regenerating data.h (or recompiling main.c) and
        # the job builds against the previous job's params. Each job has different
        # params, so there is no useful incremental state to keep anyway. clean
        # removes only the app's data.h + build/ (the shared simbacore lib,
        # snRuntime and the sbt datagen cache live elsewhere and are preserved).
        # Name the container so it can be force-removed on timeout/abort: killing
        # the `podman run` client alone leaves the container (and its make/sbt)
        # running. The seq makes the name unique across this run's builds.
        with self.procs_lock:
            self._build_seq += 1
            name = f"batchbuild_{os.getpid()}_{self._build_seq}"
            self.build_containers.add(name)
        cmd = [
            "podman",
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            *env_args,
            "-v",
            f"{self.root}:{self.root}",
            "-w",
            self.cluster,
            CONTAINER,
            "bash",
            "-c",
            f"make -C sw/apps/{app} clean && make -C sw/apps/{app}",
        ]
        with open(blog, "wb") as f:
            p = self._spawn(cmd, f)
            try:
                return p.wait(timeout=(self.build_timeout or None))
            except subprocess.TimeoutExpired:
                f.write(b"\n[batch_run] BUILD TIMEOUT\n")
                f.flush()
                self._podman_rm(name)  # stop the container, not just the client
                self._killpg(p, signal.SIGTERM)
                p.wait()
                return 124
            finally:
                self._reap(p)
                # Always tear the container down: --rm handles the clean-exit case,
                # but on abort the client is killed and only rm -f stops the rest.
                self._podman_rm(name)
                with self.procs_lock:
                    self.build_containers.discard(name)

    def run_memsim(self, jid, elf):
        """Run the cycle-accurate memsim model on the staged .elf into
        `<jid>.memsim.log`. Host-side and fast (no license), so it runs ungated
        by the vsim semaphore. Its result is supplementary -- it never changes
        the job state (the vsim/RTL run still drives Status); the report just
        scrapes this log for the Model error/SimbaCore/Total columns. Skipped
        silently if the model binary was never built.

        Also asks memsim for a `--timeline` CSV and renders it to a per-engine
        activity + TCDM-bandwidth timeline plot (`<jid>.timeline.png`)."""
        if not os.path.exists(os.path.join(self.cluster, MEMSIM_BIN)):
            return
        log = os.path.join(self.rundir, jid + ".memsim.log")
        csv = os.path.join(self.rundir, jid + ".timeline.csv")
        with open(log, "w") as f:
            p = self._spawn([MEMSIM_BIN, elf, "--timeline", csv], f)
            try:
                p.wait(timeout=(self.timeout or None))
            except subprocess.TimeoutExpired:
                self._killpg(p, signal.SIGTERM)
                p.wait()
                f.write("\n[batch_run] MEMSIM TIMEOUT\n")
            finally:
                self._reap(p)
        self.make_timeline_plot(jid, csv)

    def make_timeline_plot(self, jid, csv):
        """Render the timeline CSV to `<jid>.timeline.png` via sim/plot_timeline.py.
        Best-effort (needs matplotlib): a missing CSV (e.g. memsim timed out) or any
        plotter failure is non-fatal and never changes the job state, exactly like the
        supplementary memsim log."""
        plotter = os.path.join(self.cluster, "sim", "plot_timeline.py")
        if not os.path.exists(csv) or not os.path.exists(plotter):
            return
        png = os.path.splitext(csv)[0] + ".png"  # <jid>.timeline.png
        # Hand the plotter the real app + overrides so the title is readable
        # `key=value` params, not the crammed underscore tag from the filename.
        info = self.status["jobs"].get(jid, {})
        cmd = [sys.executable, plotter, "--csv", csv, "-o", png]
        if info.get("app"):
            cmd += ["--name", info["app"]]
        overrides = info.get("overrides") or {}
        if overrides:
            cmd += ["--params", "   ".join(f"{k}={v}" for k, v in sorted(overrides.items()))]
        with open(os.path.join(self.rundir, jid + ".plot.log"), "w") as f:
            p = self._spawn(cmd, f)
            try:
                p.wait(timeout=(self.timeout or 120))
            except subprocess.TimeoutExpired:
                self._killpg(p, signal.SIGTERM)
                p.wait()
            finally:
                self._reap(p)

    def run_sim(self, jid, elf):
        """Run the memsim model + one vsim; the vsim is gated by the parallelism
        semaphore. (Cached/force:false jobs are skipped before they reach here.)

        The staged .elf is only needed for the duration of these runs, so it is
        deleted afterwards (pass/fail/timeout/abort) to keep the out folder small."""
        log = os.path.join(self.rundir, jid + ".log")
        try:
            if not self.aborted.is_set():
                self.run_memsim(jid, elf)
            with self.sim_sem:
                if self.aborted.is_set():
                    return
                self.set_state(jid, "running")
                with open(log, "w") as f:
                    p = self._spawn([VSIM_BIN, elf], f)
                    try:
                        rc = p.wait(timeout=(self.timeout or None))
                    except subprocess.TimeoutExpired:
                        self._killpg(p, signal.SIGTERM)
                        p.wait()
                        f.write("\n[batch_run] TIMEOUT\n")
                        self.set_state(jid, "timeout")
                        return
                    finally:
                        self._reap(p)
                if not self.aborted.is_set():
                    self.set_state(jid, "done", rc=rc)
        finally:
            try:
                os.remove(elf)
            except OSError:
                pass

    def _build_order(self):
        """Round-robin across apps: appA[0], appB[0], ..., appA[1], appB[1], ...

        Builds are inherently serial (shared chisel-ssm/sbt datagen and per-app
        build dir), so interleaving lets each program's FIRST run build -- and
        therefore start simulating -- as early as possible, instead of one app
        monopolising the builder and starving the others."""
        order = []
        for i in range(max((len(lane) for lane in self.jobs), default=0)):
            for lane in self.jobs:
                if i < len(lane):
                    order.append(lane[i])
        return order

    def _process_job(self, job):
        """Build one job, stage its elf, and dispatch its sim; or, for a force:false
        cached job, just re-run the fast memsim. The make + elf-stage run under the
        app's exclusive lock so concurrent same-app builds can't clobber each other's
        generated/data dir or build/<app>.elf."""
        if self.aborted.is_set():
            return
        jid, app = job["id"], job["app"]
        elf_cache = os.path.join(os.path.dirname(self.rundir), ".elf_cache")
        if job["cached"]:
            # force:false reuses the EXPENSIVE vsim, but memsim is sub-second host-side, so never
            # cache it: re-run it every batch on the persisted .elf -> the Model columns track the
            # current binary. No rebuild, no vsim. (Bootstrap the .elf_cache with `--remodel`.)
            cached_elf = os.path.join(elf_cache, jid + ".elf")
            if os.path.exists(cached_elf):
                self.run_memsim(jid, cached_elf)
            return
        self.set_state(jid, "building")
        params_path = self.write_temp_params(job)
        staged = os.path.join(self.rundir, jid + ".elf")
        with self._app_lock(app):
            if self.aborted.is_set():
                return
            rc = self.build(app, params_path, os.path.join(self.rundir, jid + ".build.log"))
            # Stage the elf while still holding the lock, before another same-app
            # build can overwrite build/<app>.elf.
            if rc == 0:
                shutil.copy2(self.built_elf(app), staged)
        if rc != 0:
            # On abort the build was killed: leave it queued, not build_failed.
            self.set_state(jid, "queued" if self.aborted.is_set() else "build_failed", rc=rc)
            return
        # Persist it (keyed on the stable job id) so future force:false runs can re-run the fast
        # memsim model on it without rebuilding -> the Model columns never go stale.
        os.makedirs(elf_cache, exist_ok=True)
        shutil.copy2(staged, os.path.join(elf_cache, jid + ".elf"))
        # Dispatch the sim asynchronously and move on to the next build.
        t = threading.Thread(target=self.run_sim, args=(jid, staged), daemon=True)
        with self.sim_threads_lock:
            self.sim_threads.append(t)
        t.start()

    def build_worker(self):
        """Two concurrent build lanes (see the module docstring's Scheduling note):
        a serial datagen lane for not-yet-cached jobs (one sbt at a time) and a
        `build_parallel`-wide pool for cache-hit compiles, so one slow datagen no
        longer blocks dozens of already-cached builds. Returns once every build is
        done (each having dispatched its own sim)."""
        cached, ready, needs = [], [], []
        for job in self._order:
            if job["cached"]:
                cached.append(job)
            elif self._datagen_ready(job):
                ready.append(job)
            else:
                needs.append(job)

        # Fast pool: cheap memsim-only cached jobs + cache-hit compiles (no sbt).
        pool_q = queue.Queue()
        for job in cached + ready:
            pool_q.put(job)

        def pool_worker():
            while not self.aborted.is_set():
                try:
                    job = pool_q.get_nowait()
                except queue.Empty:
                    return
                self._process_job(job)

        def datagen_worker():
            # Serial: at most one sbt runs at a time across the whole batch.
            for job in needs:
                if self.aborted.is_set():
                    return
                self._process_job(job)

        workers = [threading.Thread(target=pool_worker, daemon=True) for _ in range(max(1, self.build_parallel))]
        workers.append(threading.Thread(target=datagen_worker, daemon=True))
        for w in workers:
            w.start()
        for w in workers:
            w.join()

    # --- driver ---
    def run(self):
        # Cancelling the orchestrator (Ctrl-C, kill, or tmux kill-pane/SIGHUP)
        # tears down every build/vsim process group.
        def on_signal(signum, _frame):
            raise KeyboardInterrupt

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, on_signal)

        self._order = self._build_order()
        builder = threading.Thread(target=self.build_worker, daemon=True)
        builder.start()

        interrupted = False
        try:
            # Main thread owns the terminal: render live until everything done.
            while True:
                self._render()
                with self.sim_threads_lock:
                    sims = list(self.sim_threads)
                sims_done = all(not t.is_alive() for t in sims)
                if not builder.is_alive() and sims_done:
                    break
                time.sleep(2)
        except KeyboardInterrupt:
            interrupted = True
            print("\nCancelled -- terminating all build/vsim processes...")
        finally:
            self.terminate_all()

        with self.status_lock:
            self.status["complete"] = True
            self._write_status()
        self._render()
        if interrupted:
            print("\nBatch run cancelled. All child processes terminated.")
            print(f"  Logs:   {self.rundir}")
            print(f"  Report: {os.path.join(self.report_dir, 'report.md')}")
            return
        print("\nBatch run complete.")
        print(f"  Logs:   {self.rundir}")
        print(f"  Report: {os.path.join(self.report_dir, 'report.md')}")
        kept, cached = self.no_redo_summary
        if cached:
            print(f"  force:false: ran {len(kept)} job(s); fully skipped {len(cached)} cached "
                  f"job(s) (no build/vsim/memsim -- kept stored row).")
            for jid in cached:
                print(f"    skipped {jid}")

    def _render(self):
        # Merge this batch run's current results into the single persistent report,
        # then mirror it to the terminal.
        batch_run_report.merge_run_into_report(self.report_dir, self.rundir)
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(batch_run_report.render_report(self.report_dir))
        sys.stdout.flush()

    def _probe_elf_config(self, elf):
        """Read the (seqLen, dModel) an elf was built for, straight out of memsim's --acc line, so a
        build/<app>.elf can be matched to the right job without trusting the (churned) build dir."""
        try:
            env = {**os.environ, "MEMSIM_ACC": "1"}
            out = subprocess.run([MEMSIM_BIN, elf], cwd=self.cluster, env=env,
                                 capture_output=True, text=True, timeout=60).stderr
        except Exception:
            return None
        m = re.search(r"seqLen=(\d+) dModel=(\d+)", out)
        return (m.group(1), m.group(2)) if m else None

    def remodel(self):
        """--remodel: re-run ONLY the fast memsim model on every cached/buildable elf and refresh the
        report's Model columns, reusing the stored vsim (no build, no vsim). Seeds .elf_cache from each
        app's current build/<app>.elf, matched to a job by the config the elf embeds."""
        elf_cache = os.path.join(os.path.dirname(self.rundir), ".elf_cache")
        os.makedirs(elf_cache, exist_ok=True)
        jobs = [j for lane in self.jobs for j in lane]
        # Mark EVERY job cached up front: the merge then keeps each stored row's vsim verbatim and only
        # refreshes the Model columns for jobs we actually re-memsim below (else a non-cached job with no
        # fresh log this run would get its existing row blanked).
        for j in jobs:
            self.status["jobs"][j["id"]]["cached"] = True
            self.status["jobs"][j["id"]]["state"] = self.status["jobs"][j["id"]].get("state", "done")
        by_app = {}
        for j in jobs:
            by_app.setdefault(j["app"], []).append(j)
        # Seed: probe each app's build elf once; cache it under the matching job's id.
        for app, applist in by_app.items():
            be = self.built_elf(app)
            if not os.path.exists(be):
                continue
            cfg = self._probe_elf_config(be)
            if not cfg:
                continue
            # Only seed when EXACTLY ONE job matches the elf's (seqLen, dModel): otherwise we can't tell
            # which config the single build/ elf actually is -> don't risk caching it under the wrong id.
            matches = [j for j in applist if (str(j.get("seqLen")), str(j.get("dModel"))) == cfg]
            if len(matches) != 1:
                continue
            dst = os.path.join(elf_cache, matches[0]["id"] + ".elf")
            if not os.path.exists(dst):
                shutil.copy2(be, dst)
        # memsim every job that now has a cached elf; mark it cached so the merge keeps the stored vsim.
        n = 0
        for j in jobs:
            jid = j["id"]
            ce = os.path.join(elf_cache, jid + ".elf")
            if not os.path.exists(ce):
                continue
            self.status["jobs"][jid]["cached"] = True
            self.status["jobs"][jid]["state"] = self.status["jobs"][jid].get("state", "done")
            self.run_memsim(jid, ce)
            n += 1
        self._write_status()
        self._render()
        print(f"\n--remodel: re-ran memsim on {n} cached elf(s); report Model columns refreshed (vsim reused).")
        print(f"  Report: {os.path.join(self.report_dir, 'report.md')}")

    def prune(self):
        """--prune: drop report.json rows whose job id is no longer in the current config -- dead
        configs left over from renamed params (e.g. a new key changing the app__tag). The merge keeps
        such rows forever by design ('prune by hand'); this is that hand. Backs up report.json first."""
        cur_ids = {j["id"] for lane in self.jobs for j in lane}
        report_path = os.path.join(self.report_dir, batch_run_report.REPORT_JSON)
        rep = batch_run_report._read_json(report_path) or {"jobs": {}}
        jobs = rep.get("jobs", {})
        orphans = [jid for jid in jobs if jid not in cur_ids]
        if not orphans:
            print("--prune: no orphaned rows; nothing to do.")
            return
        shutil.copy2(report_path, report_path + ".bak")
        for jid in orphans:
            del jobs[jid]
        batch_run_report._write_json_atomic(report_path, rep)
        with open(os.path.join(self.report_dir, batch_run_report.REPORT_MD), "w") as f:
            f.write(batch_run_report.render_report(self.report_dir))
        print(f"--prune: removed {len(orphans)} orphaned row(s); kept {len(jobs)}. Backup: {report_path}.bak")


def main():
    ap = argparse.ArgumentParser(description="Run a batch of simulations")
    ap.add_argument(
        "config", nargs="?", default=None, help="batch-run config (.hjson); default: <repo-root>/batch_run_config.hjson"
    )
    ap.add_argument(
        "--no-redo",
        action="store_true",
        help="force every job to force:false: fully skip configs that already produced a "
        "result (PASS or errors) -- no build, no vsim, no memsim; the report keeps their last "
        "stored row. New configs and ones whose last run was build_fail/timeout/no_result still "
        "run. Overrides per-config `force`.",
    )
    ap.add_argument(
        "--remodel",
        action="store_true",
        help="re-run ONLY the fast memsim model on every cached/buildable elf and refresh the report's "
        "Model columns (reusing the stored vsim). No build, no vsim. Seeds .elf_cache from the apps' "
        "current build/ elfs. Use after changing the memsim binary to refresh numbers in seconds.",
    )
    ap.add_argument(
        "--prune",
        action="store_true",
        help="drop report.json rows whose config is no longer in the batch config (dead 'orphan' rows "
        "left over from renamed params), then re-render. Backs up report.json first. No build/vsim/memsim.",
    )
    args = ap.parse_args()
    br = BatchRun(args.config, no_redo=args.no_redo, skip_lock=args.remodel or args.prune)
    if args.prune:
        br.prune()
    elif args.remodel:
        br.remodel()
    else:
        br.run()


if __name__ == "__main__":
    main()
