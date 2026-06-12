#!/bin/bash
set -u
cd /esat/micas-lapserv11/users/rgeens/snax_cluster/target/snitch_cluster
ROOT=$(git rev-parse --show-toplevel)
POD="podman run --rm -i -v $ROOT:$ROOT -w $(pwd) ghcr.io/kuleuven-micas/snax:main"
SUM=_t/finalval_summary.txt
: > $SUM
run() {
  local L=$1 NL=$2 NS=$3
  printf '{\n  seqLen: %s\n  dModel: 96\n  dtRank: 24\n  nb_tiles: 8\n  nb_l_tiles: %s\n  nb_slots: %s\n  bc_pad_banks: 0\n}\n' "$L" "$NL" "$NS" > sw/apps/suc-async/data/params_in.hjson
  $POD make -C sw/apps/suc-async clean-data >/dev/null 2>&1
  if ! $POD make -C sw/apps/suc-async > _t/fv_${L}_${NL}_${NS}.log 2>&1; then
    echo "$L/$NL/$NS BUILDFAIL" | tee -a $SUM; tail -6 _t/fv_${L}_${NL}_${NS}.log | tee -a $SUM; return 1
  fi
  ./bin/snitch_cluster.vsim sw/apps/suc-async/build/suc-async.elf &> tmp-suc-async.log
  local res; res=$(grep -E "Test suc-async:" tmp-suc-async.log | sed 's/^# //')
  echo "$L/$NL/$NS  ::  ${res:-NO RESULT}" | tee -a $SUM
}
run 512 8 4   # fast confirm cleaned main.c still passes
run 8192 128 4   # benchmark target (slow)
echo "=== FINALVAL DONE ===" | tee -a $SUM
