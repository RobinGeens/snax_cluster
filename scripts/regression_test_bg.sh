#!/usr/bin/env bash
set -euo pipefail

# Background launcher for regression tests, plus an inline watchdog that reaps
# every child if the run dies without cleanup (kill -9, OOM) — no make/vsim/
# container may ever outlive a dead run.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${ROOT_DIR}/regression_test_out"
mkdir -p "${OUTPUT_ROOT}"
BG_LOG="${OUTPUT_ROOT}/background_process.log"
PIDS_FILE="${OUTPUT_ROOT}/rt_children.pids"

# Fail fast (in the foreground) if a batch or regression run is already active;
# regression_test.sh takes this same lock for the whole run.
if ! flock -n "${ROOT_DIR}/.batch_run.lock" true; then
  echo "ERROR: a batch/regression run is already active (lock: ${ROOT_DIR}/.batch_run.lock)." >&2
  echo "Concurrent runs overload the server; wait for it to finish or stop it first." >&2
  exit 1
fi

# Run the regression tests in their own session and capture any errors/output
setsid nohup bash "${ROOT_DIR}/scripts/regression_test.sh" >"${BG_LOG}" 2>&1 </dev/null &
RT_PID=$!
echo "${RT_PID}" > "${OUTPUT_ROOT}/rt.pid"

# Watchdog: waits until the run is gone, then kills every process group recorded
# in the pids file that still contains one of our tools (guards against PID
# reuse) and removes the run's build containers. A clean exit reaps its own
# children via the traps in regression_test.sh, so this finds nothing to do.
setsid nohup bash -c '
  rt_pid="$1"; pids_file="$2"
  while kill -0 "${rt_pid}" 2>/dev/null; do sleep "${RT_WATCHDOG_POLL:-30}"; done
  sleep 5 # on a clean death, let the run finish its own EXIT trap first
  for sig in TERM KILL; do
    if [ -f "${pids_file}" ]; then
      while IFS= read -r pg; do
        [ -n "${pg}" ] && [ "${pg}" -gt 1 ] 2>/dev/null || continue
        if pgrep -g "${pg}" -a 2>/dev/null | grep -qE "vsim|snitch_cluster|build_sim|podman|timeout|make|bender"; then
          echo "sweeping leftover process group ${pg} (${sig})"
          pkill "-${sig}" -g "${pg}" 2>/dev/null || true
        fi
      done < "${pids_file}"
    fi
    sleep 5
  done
  podman ps -a --format "{{.Names}}" 2>/dev/null | grep "^rtbuild_${rt_pid}_" | while read -r name; do
    echo "removing leftover build container ${name}"
    podman rm -f "${name}" >/dev/null 2>&1 || true
  done
  echo "watchdog done"
' rt-watchdog "${RT_PID}" "${PIDS_FILE}" >"${OUTPUT_ROOT}/watchdog.log" 2>&1 </dev/null &

echo "Regression tests started in background (PID ${RT_PID})." >&2
echo "  Log:  ${BG_LOG}" >&2

# Relay the run's summary/build-log paths: the run has no controlling terminal
# (setsid), so it can only print them into BG_LOG. Echo them here once they appear.
for _ in $(seq 1 20); do
  grep -q '^Build log:' "${BG_LOG}" 2>/dev/null && break
  sleep 0.5
done
grep -E '^(Summary file|Build log):' "${BG_LOG}" >&2 || true
