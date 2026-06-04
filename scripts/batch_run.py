#!/usr/bin/env python3
"""Run a whole batch of simulations from one config file.

Reads an hjson config listing apps and, per app, a list of param-override sets.
For each (app, override-set) it: writes the merged params_in.hjson, builds the
app's .elf (in the podman container), stages the .elf to a unique path, then
launches the vsim. Each run gets its own log file.

Scheduling:
  * Builds are serialized globally (one `make` at a time) -- params_in.hjson and
    the build dir are shared per app, so this is the natural serialization point
    and also avoids podman/make races across apps.
  * Sims run concurrently, bounded by `max_parallel` (the OOM knob).
  * A lane (one app's override list) does not wait for a run's sim to finish
    before building the next override set -- the next run starts as soon as the
    previous build is done.
  * Different apps' lanes run concurrently => different programs in parallel.

Results are merged into ONE persistent report at the repo root (<root>/report.md
+ report.json). The merge is self-cleaning per app: rerunning an app wipes its
old rows and writes only this batch run's jobs, so stale/contaminated rows are
replaced; apps not in this batch run are kept (accumulation by app). A Batch run
column records which timestamped batch run each number came from. View it
separately with:
    python3 scripts/batch_run_report.py            # run from the repo root

Usage (run from anywhere in the repo):
    python3 scripts/batch_run.py                       # uses <root>/batch_run_config.hjson
    python3 scripts/batch_run.py path/to/other.hjson
"""
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import hjson

import batch_run_report

CONTAINER = "ghcr.io/kuleuven-micas/snax:main"
VSIM_BIN = "bin/snitch_cluster.vsim"


def repo_root():
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    return out.decode().strip()


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
    def __init__(self, config_path=None):
        self.root = repo_root()
        self.cluster = os.path.join(self.root, "target", "snitch_cluster")
        # Only ONE batch run may run at a time: concurrent batch runs share each
        # app's params_in.hjson, the per-app build dir, the chisel-ssm/sbt datagen
        # and the root report.json -- running two corrupts builds and results. Take
        # an exclusive lock that the OS releases automatically if we die.
        self._lock_fd = open(os.path.join(self.root, ".batch_run.lock"), "w")
        try:
            fcntl.lockf(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            held = open(os.path.join(self.root, ".batch_run.lock")).read().strip()
            sys.exit(f"Another batch run is already running (PID {held or '?'}). "
                     f"Refusing to start a second one -- it would corrupt the "
                     f"shared params_in.hjson and report. Lock: {self.root}/.batch_run.lock")
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
        # Per-run wall-clock cap in seconds; 0 disables (default: none).
        self.timeout = int(self.cfg.get("timeout", 0))

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
        self.procs = {}                       # pid -> Popen
        self.procs_lock = threading.Lock()
        self.aborted = threading.Event()

        self.jobs = self._build_job_list()   # ordered list per lane
        self.status = {
            "rundir": self.rundir,
            "config_path": self.config_path,
            "started": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "max_parallel": self.max_parallel,
            "complete": False,
            "jobs": {},
        }
        for lane in self.jobs:
            for job in lane:
                self.status["jobs"][job["id"]] = {
                    "app": job["app"], "tag": job["tag"],
                    "params": job["all_params"],
                    "seqLen": job["seqLen"], "dModel": job["dModel"],
                    "n_tiles": job["n_tiles"], "log": job["id"] + ".log",
                    "state": "queued", "rc": None,
                }
        self._write_status()

    def _build_job_list(self):
        """Return a list of lanes; each lane is an ordered list of job dicts."""
        lanes = []
        seen_ids = set()
        for entry in self.cfg["runs"]:
            app = entry["app"]
            # Base params (working-tree params_in.hjson) so the report can show
            # effective seqLen/dModel/n_tiles for every job, default ones included.
            base = {}
            bpath = self.params_path(app)
            if os.path.exists(bpath):
                with open(bpath) as f:
                    base = hjson.load(f)
            param_sets = entry.get("params") or [{}]
            lane = []
            for overrides in param_sets:
                overrides = dict(overrides)
                eff = {**base, **overrides}
                tag = make_tag(overrides)
                jid = f"{app}__{tag}"
                n = 2
                while jid in seen_ids:   # disambiguate duplicate override sets
                    jid = f"{app}__{tag}__{n}"
                    n += 1
                seen_ids.add(jid)
                lane.append({"id": jid, "app": app, "tag": tag,
                             "overrides": overrides, "all_params": dict(eff),
                             "seqLen": _pick(eff, "seqLen", "dim0"),
                             "dModel": _pick(eff, "dModel", "dim1"),
                             "n_tiles": _pick(eff, "nb_tiles", "n_tiles")})
            lanes.append(lane)
        return lanes

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
        return os.path.join(self.cluster, "sw", "apps", app, "data",
                            "params_in.hjson")

    def built_elf(self, app):
        return os.path.join(self.cluster, "sw", "apps", app, "build",
                            app + ".elf")

    # --- build / run primitives ---
    def write_params(self, app, overrides):
        path = self.params_path(app)
        if not os.path.exists(path):
            if overrides:
                raise FileNotFoundError(
                    f"{app} has no params_in.hjson but overrides were given: "
                    f"{overrides}")
            return  # params-less app (e.g. nop) with empty overrides: nothing to do
        with open(path) as f:
            params = hjson.load(f)
        params.update(overrides)
        with open(path, "w") as f:
            hjson.dump(params, f)

    # --- child process management (every child runs in its own process group
    #     so cancelling the orchestrator tears down the whole bash+vsim tree) ---
    def _spawn(self, cmd, stdout):
        p = subprocess.Popen(cmd, cwd=self.cluster, stdout=stdout,
                             stderr=subprocess.STDOUT, start_new_session=True)
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

    def build(self, app, blog):
        cmd = [
            "podman", "run", "--rm", "-i",
            "-v", f"{self.root}:{self.root}", "-w", self.cluster,
            CONTAINER, "make", "-C", f"sw/apps/{app}",
        ]
        with open(blog, "wb") as f:
            p = self._spawn(cmd, f)
            try:
                return p.wait()
            finally:
                self._reap(p)

    def run_sim(self, jid, elf):
        """Run one vsim; gated by the parallelism semaphore.

        The staged .elf is only needed for the duration of this sim, so it is
        deleted afterwards (pass/fail/timeout/abort) to keep the out folder small."""
        log = os.path.join(self.rundir, jid + ".log")
        try:
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

    def build_worker(self):
        """Single serial builder: build each job in fair order, dispatch its sim."""
        for job in self._order:
            if self.aborted.is_set():
                return
            jid, app = job["id"], job["app"]
            self.set_state(jid, "building")
            self.write_params(app, job["overrides"])
            rc = self.build(app, os.path.join(self.rundir, jid + ".build.log"))
            if rc != 0:
                # On abort the build was killed: leave it queued, not build_failed.
                self.set_state(jid, "queued" if self.aborted.is_set() else "build_failed",
                               rc=rc)
                continue
            # Stage the elf before the next build overwrites build/<app>.elf.
            staged = os.path.join(self.rundir, jid + ".elf")
            shutil.copy2(self.built_elf(app), staged)
            # Dispatch the sim asynchronously and move on to the next build.
            t = threading.Thread(target=self.run_sim, args=(jid, staged),
                                 daemon=True)
            with self.sim_threads_lock:
                self.sim_threads.append(t)
            t.start()

    # --- driver ---
    def run(self):
        # Back up each app's params so the working tree is restored afterwards.
        apps = {job["app"] for lane in self.jobs for job in lane}
        backups = {}
        for app in apps:
            p = self.params_path(app)
            if os.path.exists(p):  # params-less apps (e.g. nop) have nothing to restore
                backups[app] = open(p).read()

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
            for app, content in backups.items():
                with open(self.params_path(app), "w") as f:
                    f.write(content)

        with self.status_lock:
            self.status["complete"] = True
            self._write_status()
        self._render()
        if interrupted:
            print("\nBatch run cancelled. All child processes terminated.")
            print(f"  Logs:   {self.rundir}")
            print(f"  Report: {os.path.join(self.report_dir, 'report.md')}")
            print("Restored each app's params_in.hjson to its pre-run state.")
            return
        print("\nBatch run complete.")
        print(f"  Logs:   {self.rundir}")
        print(f"  Report: {os.path.join(self.report_dir, 'report.md')}")
        print("Restored each app's params_in.hjson to its pre-run state.")

    def _render(self):
        # Merge this batch run's current results into the single persistent report,
        # then mirror it to the terminal.
        batch_run_report.merge_run_into_report(self.report_dir, self.rundir)
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(batch_run_report.render_report(self.report_dir))
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="Run a batch of simulations")
    ap.add_argument("config", nargs="?", default=None,
                    help="batch-run config (.hjson); default: <repo-root>/batch_run_config.hjson")
    args = ap.parse_args()
    BatchRun(args.config).run()


if __name__ == "__main__":
    main()
