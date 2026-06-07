# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Robin Geens <robin.geens@esat.kuleuven.be>

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR ?= $(realpath $(MK_DIR))
CURR_DIR ?= $(DATA_DIR)
CLUSTER_DIR := $(abspath $(CURR_DIR)/../../../../)

DATAGEN_PY := $(DATA_DIR)/datagen.py
DATAGEN_DEPS := $(CURR_DIR)/../../main/data/datagen_base.py
DATAGEN_DEPS += $(CURR_DIR)/../../main/data/datagen_cli.py
DATA_H     := $(DATA_DIR)/data.h


WORKLOAD_PARAMS := $(or $(PARAMS_IN),$(CURR_DIR)/params_in.hjson)

APP_NAME = is-osgemm-tiled-async

.PHONY: clean-data clean
clean-data:
	rm -f $(DATA_H)
clean: clean-data

CHISEL_SSM := $(shell bender path chisel-ssm 2>/dev/null)
ifneq ($(CHISEL_SSM),)

SBT_GEN_DIR := $(CHISEL_SSM)/generated/data/$(APP_NAME)
DATA_CFG    := $(SBT_GEN_DIR)/params.hjson

# Read workload params from hjson (strip comments and quotes)
SEQ_LEN := $(shell grep -oP '"?seqLen"?\s*:\s*\K[0-9]+' $(WORKLOAD_PARAMS))
D_MODEL := $(shell grep -oP '"?dModel"?\s*:\s*\K[0-9]+' $(WORKLOAD_PARAMS))
D_INNER := $(shell grep -oP '"?dInner"?\s*:\s*\K[0-9]+' $(WORKLOAD_PARAMS))

# Cache key based on dimensions (ring params only affect the Python datagen)
CACHE_ARGS     := seqLen=$(SEQ_LEN) dModel=$(D_MODEL) dInner=$(D_INNER)
DATAGEN_CACHE_DIR := $(CLUSTER_DIR)/.datagen_cache/$(APP_NAME)

.PHONY: clean-cache
clean-cache:
	rm -rf $(DATAGEN_CACHE_DIR)

# Runs BOTH Scala generators (OSGEMM mode 3 + ISGEMM mode 4) into the same output dir,
# then the Python datagen merges both into a single data.h.
$(DATA_H): $(WORKLOAD_PARAMS) $(DATAGEN_PY) $(DATAGEN_DEPS)
	@set -e; \
	CACHE_KEY=$$(echo "$(CACHE_ARGS)" | md5sum | cut -d' ' -f1); \
	CACHED="$(DATAGEN_CACHE_DIR)/$$CACHE_KEY.tar"; \
	if [ -f "$$CACHED" ]; then \
		echo "[DATAGEN CACHE] Hit ($(APP_NAME)) - restoring sbt output"; \
		mkdir -p "$(SBT_GEN_DIR)"; \
		tar xf "$$CACHED" -C "$(SBT_GEN_DIR)"; \
	else \
		echo "[DATAGEN CACHE] Miss ($(APP_NAME)) - running sbt (OSGEMM + ISGEMM)"; \
		echo "  OSGEMM: dim0=$(SEQ_LEN) dim1=$(D_MODEL) dim2=$(D_INNER)"; \
		cd $(CHISEL_SSM) && sbt "test:runMain datagen.DataGeneratorOSGEMM dim0=$(SEQ_LEN) dim1=$(D_MODEL) dim2=$(D_INNER) name=$(APP_NAME)"; \
		echo "  ISGEMM: dim0=$(SEQ_LEN) dim1=$(D_INNER) dim2=$(D_MODEL)"; \
		cd $(CHISEL_SSM) && sbt "test:runMain datagen.DataGeneratorISGEMM dim0=$(SEQ_LEN) dim1=$(D_INNER) dim2=$(D_MODEL) name=$(APP_NAME)"; \
		if [ ! -f "$(DATA_CFG)" ]; then \
			echo "[DATAGEN CACHE] ERROR: sbt produced no $(DATA_CFG) for $(APP_NAME) — check params_in.hjson" >&2; \
			exit 1; \
		fi; \
		mkdir -p "$(DATAGEN_CACHE_DIR)"; \
		tar cf "$$CACHED" -C "$(SBT_GEN_DIR)" .; \
		echo "[DATAGEN CACHE] Stored $(APP_NAME) -> $$CACHE_KEY"; \
	fi
	@echo "Generating data.h from $(DATA_CFG)"
	@$(DATAGEN_PY) --swcfg $(DATA_CFG) > $@

endif
