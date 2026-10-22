#!/usr/bin/env bash
set -euo pipefail

# Run all specified SNAX-level tests and collect results.
# Clones the repo into a temporary directory so the current build state is untouched.
# Results and logs are written to regression_test_out/ in the original repo.


TARGET_DIR="target/snitch_cluster"
CFG_OVERRIDE="cfg/snax_simbacore_cluster.hjson"
VSIM_BIN="bin/snitch_cluster.vsim"

# Caps (env-overridable; 0 disables the resource limits).
SIM_TIMEOUT="${RT_SIM_TIMEOUT:-43200}"      # per-test wall clock [s]
BUILD_TIMEOUT="${RT_BUILD_TIMEOUT:-21600}"  # build wall clock [s]
VMEM_GB="${RT_VMEM_GB:-48}"                 # per-sim address-space cap [GiB]
FSIZE_GB="${RT_FSIZE_GB:-8}"                # per-sim max single-file size [GiB] (.dasm traces)

ORIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${ORIG_ROOT}/regression_test_out"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_ROOT}"

# --- Single-runner lock, shared with batch_run.py ---
# A concurrent batch + regression run is what overloaded the server: ~18 vsims,
# two build pipelines and unbounded traces at once. Refuse to double-book.
LOCK_FILE="${ORIG_ROOT}/.batch_run.lock"
exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
  echo "ERROR: another batch/regression run is already active (lock: ${LOCK_FILE})." >&2
  echo "Concurrent runs overload the server; wait for it to finish or stop it first." >&2
  exit 1
fi
printf 'regression_test %s\n' "$$" >&8

# Get commit hash from the original repo
if command -v git >/dev/null 2>&1 && git -C "${ORIG_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  COMMIT_HASH="$(git -C "${ORIG_ROOT}" rev-parse --short HEAD 2>/dev/null || echo no-git)"
  COMMIT_MSG="$(git -C "${ORIG_ROOT}" log -1 --pretty=%s 2>/dev/null || echo "")"
else
  COMMIT_HASH="no-git"
  COMMIT_MSG="no-git"
fi

RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}-${COMMIT_HASH}"
mkdir -p "${RUN_DIR}"
SUMMARY_FILE="${RUN_DIR}/summary.log"
BUILD_LOG="${RUN_DIR}/build.log"

# Create and print files so we can easily open them
: > "${SUMMARY_FILE}"
: > "${BUILD_LOG}"
{
  echo ""
  echo "Summary file: ${SUMMARY_FILE}"
  echo "Build log:    ${BUILD_LOG}"
} >&2

# --- Child tracking + teardown ---
# Every spawned child (build, sims) runs in its own session/process group and its
# pgid is recorded here; rt_watchdog.sh reads the same file if this script dies
# without cleanup (kill -9, OOM).
PIDS_FILE="${OUTPUT_ROOT}/rt_children.pids"
: > "${PIDS_FILE}"
CONTAINER_PREFIX="rtbuild_$$"

# Kill every recorded child process group: TERM, grace, then KILL. A group is only
# signalled while it still contains one of our tools (guards against PID reuse).
kill_children() {
  local sig pg
  for sig in TERM KILL; do
    if [ -f "${PIDS_FILE}" ]; then
      while IFS= read -r pg; do
        [ -n "${pg}" ] && [ "${pg}" -gt 1 ] 2>/dev/null || continue
        if pgrep -g "${pg}" -a 2>/dev/null | grep -qE 'vsim|snitch_cluster|build_sim|podman|timeout|make|bender'; then
          pkill "-${sig}" -g "${pg}" 2>/dev/null || true
        fi
      done < "${PIDS_FILE}"
    fi
    if [ "${sig}" = TERM ]; then sleep 3; fi
  done
  # Build containers outlive their podman client; remove them by name.
  podman ps -a --format '{{.Names}}' 2>/dev/null | grep "^${CONTAINER_PREFIX}_" \
    | xargs -r -n1 podman rm -f >/dev/null 2>&1 || true
  return 0
}

# --- Create a temporary clone for a clean build ---
# Place it under OUTPUT_ROOT so podman can bind-mount it inside the container.
TMPDIR_ROOT="$(mktemp -d -p "${OUTPUT_ROOT}" tmp.XXXXXX)"
WORK_DIR="${TMPDIR_ROOT}/snax_cluster"

cleanup() {
  kill_children
  echo "Cleaning up temporary build directory: ${TMPDIR_ROOT}" >&2
  # Best-effort: NFS can briefly hold files of just-killed sims open (.nfs*).
  # Never die or hang here — a failed rm must not cost results or leak processes.
  if ! rm -rf "${TMPDIR_ROOT}" 2>/dev/null; then
    sleep 5
    rm -rf "${TMPDIR_ROOT}" 2>/dev/null \
      || echo "WARNING: could not fully remove ${TMPDIR_ROOT}; remove it manually." >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

# Children must not inherit the lock fd (8>&-): an inherited flock keeps the
# lock held by any surviving child after this script is killed.
echo "Cloning repository into ${WORK_DIR} (committed state only) ..." >&2
git clone --local --no-hardlinks "${ORIG_ROOT}" "${WORK_DIR}" 2>&1 >&2 8>&-

# Populate submodules (they are not pulled in by `git clone --local`).
echo "Initializing submodules ..." >&2
git -C "${WORK_DIR}" submodule update --init --recursive >&2 8>&-

echo "Temporary clone ready at ${WORK_DIR}" >&2

# Build in the temporary clone: own session/pgroup (recorded for teardown), hard
# wall-clock cap, and named podman containers so an orphaned build can always be
# stopped (killing the podman client alone leaves make/sbt running inside).
build_rc=0
setsid env CONTAINER_NAME_PREFIX="${CONTAINER_PREFIX}" \
  timeout -k 120 "${BUILD_TIMEOUT}" bash "${WORK_DIR}/scripts/build_sim.sh" \
  > "${BUILD_LOG}" 2>&1 < /dev/null 8>&- &
build_pid=$!
echo "${build_pid}" >> "${PIDS_FILE}"
wait "${build_pid}" || build_rc=$?

# Read the list of test programs from the Makefile
read -ra TESTS <<< "$(make -C "${WORK_DIR}/${TARGET_DIR}/sw/apps" -s list-apps)"

pushd "${WORK_DIR}/${TARGET_DIR}" >/dev/null

{ # Summary header
  echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S')"
  echo "Commit: (${COMMIT_HASH}) \"${COMMIT_MSG}\""
  chisel_ssm_commit="$(grep chisel-ssm "${WORK_DIR}/Bender.lock" -A3 | grep revision | awk '{print $2}' | head -n1)"
  echo "Chisel-SSM version: ${chisel_ssm_commit}"
  if [ "${build_rc}" -eq 0 ]; then
    echo "Build: ✅ SUCCESS"
  elif [ "${build_rc}" -eq 124 ] || [ "${build_rc}" -eq 137 ]; then
    echo "Build: ❌ TIMEOUT after ${BUILD_TIMEOUT}s"
  else
    echo "Build: ❌ FAILED (rc=${build_rc})"
  fi
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  printf "%-30s %10s %18s %18s\n" "Test Name" "Errors" "SimbaCore Cycles" "Total Cycles"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
} > "${SUMMARY_FILE}"

# --- Run all tests in parallel, each in an isolated scratch CWD ---
VSIM_ABS="$(pwd)/${VSIM_BIN}"
SCRATCH_ROOT="${TMPDIR_ROOT}/scratch"
mkdir -p "${SCRATCH_ROOT}"
JOBS="${REGRESSION_JOBS:-10}"
[ "${JOBS}" -le 0 ] && JOBS="${#TESTS[@]}"

# Run one test, then parse its log and append its summary row immediately.
run_one() {
  local name="$1"
  local elf_abs="$(pwd)/sw/apps/${name}/build/${name}.elf"
  local test_log="${RUN_DIR}/${name}.log"
  local scratch="${SCRATCH_ROOT}/${name}"
  mkdir -p "${scratch}"
  # Own session/pgroup (recorded for teardown), resource-capped: a runaway sim
  # dies alone instead of exhausting server RAM (-v) or filling the volume with
  # .dasm traces (-f).
  local rc=0 pid
  (
    cd "${scratch}"
    if [ "${VMEM_GB}" -gt 0 ]; then ulimit -S -v $((VMEM_GB * 1024 * 1024)); fi
    if [ "${FSIZE_GB}" -gt 0 ]; then ulimit -S -f $((FSIZE_GB * 2 * 1024 * 1024)); fi
    exec setsid timeout -k 60 "${SIM_TIMEOUT}" "${VSIM_ABS}" "${elf_abs}"
  ) > "${test_log}" 2>&1 < /dev/null &
  pid=$!
  echo "${pid}" >> "${PIDS_FILE}"
  wait "${pid}" || rc=$?

  # Parse error count from this test's log (124/137 = timeout kill)
  local errors="" parsed_errors
  local timed_out=0
  if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then timed_out=1; fi
  parsed_errors="$(sed -n 's/.*Finished with exit code[[:space:]]\+\([0-9]\+\).*/\1/p' "${test_log}" | tail -n1)"
  if [ -z "${parsed_errors}" ]; then
    # No app completion marker -> the program never exited
    local vsim_errors
    vsim_errors="$(sed -n 's/.*Errors:[[:space:]]\+\([0-9]\+\).*/\1/p' "${test_log}" | tail -n1)"
    if [ -n "${vsim_errors}" ] && [ "${vsim_errors}" -gt 0 ]; then
      parsed_errors="CRASH"
    else
      parsed_errors="${vsim_errors}"
    fi
  fi
  if [ "${timed_out}" -eq 1 ]; then
    errors="TIMEOUT"
  elif [ -n "${parsed_errors}" ]; then
    errors="${parsed_errors}"
  else
    # Fall back to process return code.
    errors="${rc}"
  fi

  # Parse cycle counts from this test's log
  local simbacore_cycles total_cycles
  simbacore_cycles="$(sed -n 's/.*Simbacore elapsed time:[[:space:]]\+\([0-9]\+\)[[:space:]]\+cycles.*/\1/p' "${test_log}" | tail -n1)"
  total_cycles="$(sed -n 's/.*Snitch elapsed time:[[:space:]]\+\([0-9]\+\)[[:space:]]\+cycles.*/\1/p' "${test_log}" | tail -n1)"

  # Append table row as soon as this test finishes (single-line append is atomic).
  # This must happen BEFORE any scratch cleanup: a failed rm on NFS used to abort
  # run_one under set -e and silently drop the row.
  printf "%-30s %10s %18s %18s\n" "${name}" "${errors}" "${simbacore_cycles:-N/A}" "${total_cycles:-N/A}" >> "${SUMMARY_FILE}"

  # Sweep any vsim descendants the wrapper left behind, then delete the scratch
  # (best-effort: an NFS straggler must not abort the run).
  pkill -KILL -g "${pid}" 2>/dev/null || true
  rm -rf "${scratch}" 2>/dev/null || true
}

running=0
for name in "${TESTS[@]}"; do
  run_one "${name}" 8>&- &
  running=$((running + 1))
  if [ "${running}" -ge "${JOBS}" ]; then
    wait -n || true
    running=$((running - 1))
  fi
done
wait || true

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "${SUMMARY_FILE}"

popd >/dev/null
