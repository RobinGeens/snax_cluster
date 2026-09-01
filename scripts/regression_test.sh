#!/usr/bin/env bash
set -euo pipefail

# Run all specified SNAX-level tests and collect results.
# Clones the repo into a temporary directory so the current build state is untouched.
# Results and logs are written to regression_test_out/ in the original repo.

TARGET_DIR="target/snitch_cluster"
CFG_OVERRIDE="cfg/snax_simbacore_cluster.hjson"
VSIM_BIN="bin/snitch_cluster.vsim"

ORIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${ORIG_ROOT}/regression_test_out"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_ROOT}"

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

# Create and print file so we can easily open it
: > "${SUMMARY_FILE}"
: > "${BUILD_LOG}"
{
  echo ""
  echo "Summary file: ${SUMMARY_FILE}"
  echo "Build log:    ${BUILD_LOG}"
} > /dev/tty


# --- Create a temporary clone for a clean build ---
# Place it under OUTPUT_ROOT so podman can bind-mount it inside the container.
TMPDIR_ROOT="$(mktemp -d -p "${OUTPUT_ROOT}" tmp.XXXXXX)"
WORK_DIR="${TMPDIR_ROOT}/snax_cluster"

cleanup() {
  echo "Cleaning up temporary build directory: ${TMPDIR_ROOT}" >&2
  rm -rf "${TMPDIR_ROOT}"
}
trap cleanup EXIT

echo "Cloning repository into ${WORK_DIR} (committed state only) ..." >&2
git clone --local --no-hardlinks "${ORIG_ROOT}" "${WORK_DIR}" 2>&1 >&2

# Populate submodules (they are not pulled in by `git clone --local`).
echo "Initializing submodules ..." >&2
git -C "${WORK_DIR}" submodule update --init --recursive >&2

echo "Temporary clone ready at ${WORK_DIR}" >&2

# Build in the temporary clone
build_rc=0
if bash "${WORK_DIR}/scripts/build_sim.sh" > "${BUILD_LOG}" 2>&1; then
  :
else
  build_rc=$?
fi

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
  else
    echo "Build: ❌ FAILED (rc=${build_rc})"
  fi
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  printf "%-30s %10s %18s %18s\n" "Test Name" "Errors" "SimbaCore Cycles" "Total Cycles"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
} > "${SUMMARY_FILE}"

TIMEOUT_TESTS=()

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
  # Must not fail under set -e: timeout exits 124 when the limit is hit.
  local rc=0
  ( cd "${scratch}" && timeout -k 60 43200 "${VSIM_ABS}" "${elf_abs}" ) > "${test_log}" 2>&1 || rc=$?
  rm -rf "${scratch}"

  # Parse error count from this test's log (124 = timeout exit code)
  local errors="" parsed_errors
  local timed_out=$(( rc == 124 ))
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

  # Append table row as soon as this test finishes (single-line append is atomic)
  printf "%-30s %10s %18s %18s\n" "${name}" "${errors}" "${simbacore_cycles:-N/A}" "${total_cycles:-N/A}" >> "${SUMMARY_FILE}"
}

running=0
for name in "${TESTS[@]}"; do
  run_one "${name}" &
  running=$((running + 1))
  if [ "${running}" -ge "${JOBS}" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "${SUMMARY_FILE}"

popd >/dev/null
