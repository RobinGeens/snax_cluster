#!/usr/bin/env bash
# Bundle makefile commands to build RTL, software and simulator.
# Supports running in and outside of the SNAX container.


set -euo pipefail


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
CONTAINER_CMD="podman run --rm -i -v \"${ROOT_DIR}\":\"${ROOT_DIR}\" -w \"${ROOT_DIR}\" ghcr.io/kuleuven-micas/snax:main bash -s"


eval "${CONTAINER_CMD}" <<'IN_CONTAINER'
set -e
cd target/snitch_cluster
make clean
IN_CONTAINER

bender update --fetch

eval "${CONTAINER_CMD}" <<'IN_CONTAINER'
set -e
cd target/snitch_cluster
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson rtl-gen
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson vsim_preparation
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson sw
IN_CONTAINER

# Make simulator
cd "${ROOT_DIR}/target/snitch_cluster"
make CFG_OVERRIDE=cfg/snax_simbacore_cluster.hjson bin/snitch_cluster.vsim 



