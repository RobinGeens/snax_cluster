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

# --- Create a temporary clone for a clean build ---
TMPDIR_ROOT="$(mktemp -d)"
WORK_DIR="${TMPDIR_ROOT}/snax_cluster"

cleanup() {
  echo "Cleaning up temporary build directory: ${TMPDIR_ROOT}" >&2
  rm -rf "${TMPDIR_ROOT}"
}
trap cleanup EXIT

echo "Cloning repository into ${WORK_DIR} (committed state only) ..." >&2
git clone --local "${ORIG_ROOT}" "${WORK_DIR}" 2>&1 >&2

echo "Temporary clone ready at ${WORK_DIR}" >&2

# Build in the temporary clone
build_rc=0
if bash "${WORK_DIR}/scripts/build_sim.sh" > "${BUILD_LOG}" 2>&1; then
  :
else
  build_rc=$?
fi

# Define tests (names only; ELF at sw/apps/[name]/build/[name].elf)
declare -a TESTS=(
  "nop"
  "fft-3way"
  "main"
  "osgemm"
  "osgemm-tiled"
  "isgemm"
  "isgemm-tiled"
  "simd"
  "main-full"
  "main-tiled"
  "einfft"
  "einfft-tiled"
  "rmsnorm"
  "fft"
  "fft-tiled"
  "fft-3way-tiled"
  "streamer-burner"
  "core-burner"
  "suc-only"
)

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

for name in "${TESTS[@]}"; do
  elf_rel="sw/apps/${name}/build/${name}.elf"
  test_log="${RUN_DIR}/${name}.log"

  # Run test (12 hour timeout; SIGKILL after 60s if still running)
  timeout -k 60 43200 "${VSIM_BIN}" "${elf_rel}" > "${test_log}" 2>&1

  # Parse error count from this test's log
  rc=$?
  errors=""
  parsed_errors="$(sed -n 's/.*Finished with exit code[[:space:]]\+\([0-9]\+\).*/\1/p' "${test_log}" | tail -n1)"
  if [ -z "${parsed_errors}" ]; then
    # Fallback pattern present in some logs: "Errors: N"
    parsed_errors="$(sed -n 's/.*Errors:[[:space:]]\+\([0-9]\+\).*/\1/p' "${test_log}" | tail -n1)"
  fi
  if [ -n "${parsed_errors}" ]; then
    errors="${parsed_errors}"
  elif [ "${rc}" -eq 124 ]; then
    # If timeout (124), count as a large number.
    errors=9999
  else
    # Fall back to process return code.
    errors="${rc}"
  fi

  # Parse cycle counts from this test's log
  simbacore_cycles=""
  total_cycles=""
  parsed_simbacore="$(sed -n 's/.*Simbacore elapsed time:[[:space:]]\+\([0-9]\+\)[[:space:]]\+cycles.*/\1/p' "${test_log}" | tail -n1)"
  parsed_total="$(sed -n 's/.*Snitch elapsed time:[[:space:]]\+\([0-9]\+\)[[:space:]]\+cycles.*/\1/p' "${test_log}" | tail -n1)"
  if [ -n "${parsed_simbacore}" ]; then
    simbacore_cycles="${parsed_simbacore}"
  fi
  if [ -n "${parsed_total}" ]; then
    total_cycles="${parsed_total}"
  fi

  # Format output as table row
  simbacore_display="${simbacore_cycles:-N/A}"
  total_display="${total_cycles:-N/A}"
  printf "%-30s %10s %18s %18s\n" "${name}" "${errors}" "${simbacore_display}" "${total_display}" >> "${SUMMARY_FILE}"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "${SUMMARY_FILE}"

popd >/dev/null
