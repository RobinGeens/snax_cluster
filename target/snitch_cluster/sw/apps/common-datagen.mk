# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@esat.kuleuven.be>

CFG_OVERRIDE ?= cfg/snax_simbacore_cluster.hjson
# Path to the snitch_cluster directory (four levels up from this data dir)
CLUSTER_DIR := $(abspath $(CURR_DIR)/../../../../)

DATAGEN_PY ?= $(DATA_DIR)/datagen.py
DATAGEN_DEPS ?= $(CURR_DIR)/../../main/data/datagen_base.py
DATAGEN_DEPS += $(CURR_DIR)/../../main/data/datagen_cli.py
DATA_H     ?= $(DATA_DIR)/data.h

# Read desired workload parameters from a local params file (input)
WORKLOAD_PARAMS := $(CURR_DIR)/params_in.hjson

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H) $(EXTRA_CLEAN)

clean: clean-data


CHISEL_SSM := $(shell bender path chisel-ssm 2>/dev/null)
ifneq ($(CHISEL_SSM),)

DATA_CFG     ?= $(CHISEL_SSM)/generated/data/$(APP_NAME)/params.hjson
SBT_GEN_DIR  := $(CHISEL_SSM)/generated/data/$(APP_NAME)

# Build generator args from all top-level key:value pairs (key=value)
GENERATOR_ARGS := $(strip $(shell grep -E '^[[:space:]]*("[^"]+"|[A-Za-z0-9_]+)[[:space:]]*:' $(WORKLOAD_PARAMS) | sed -e 's://.*$$::' -e 's/[",{}]//g' -e 's/^[[:space:]]*//;s/[[:space:]]*$$//' -e 's/[[:space:]]*:[[:space:]]*/=/' | tr '\n' ' '))
GENERATOR_ARGS += name=$(APP_NAME)

# Only use L and D for data cache key
DATAGEN_CACHE_DIR := $(CLUSTER_DIR)/.datagen_cache/$(APP_NAME)
CACHE_ARGS        := $(filter-out nb_tiles% name=%,$(GENERATOR_ARGS))

.PHONY: clean-cache cache-seed

clean-cache:
	rm -rf $(DATAGEN_CACHE_DIR)

cache-seed:
	@if [ ! -d "$(SBT_GEN_DIR)" ]; then \
		echo "No sbt output for $(APP_NAME)"; exit 0; \
	fi
	@CACHE_KEY=$$(echo "$(CACHE_ARGS)" | md5sum | cut -d' ' -f1); \
	mkdir -p "$(DATAGEN_CACHE_DIR)"; \
	tar cf "$(DATAGEN_CACHE_DIR)/$$CACHE_KEY.tar" -C "$(SBT_GEN_DIR)" .; \
	echo "[DATAGEN CACHE] Seeded $(APP_NAME) → $$CACHE_KEY"

# Step 1: ensure sbt golden data exists (from cache or fresh run).
# Step 2: always run Python datagen to produce data.h (fast, encodes nb_tiles).
$(DATA_H): $(WORKLOAD_PARAMS) $(DATAGEN_PY) $(DATAGEN_DEPS)
	@CACHE_KEY=$$(echo "$(CACHE_ARGS)" | md5sum | cut -d' ' -f1); \
	CACHED="$(DATAGEN_CACHE_DIR)/$$CACHE_KEY.tar"; \
	if [ -f "$$CACHED" ]; then \
		echo "[DATAGEN CACHE] Hit ($(APP_NAME)) — restoring sbt output"; \
		mkdir -p "$(SBT_GEN_DIR)"; \
		tar xf "$$CACHED" -C "$(SBT_GEN_DIR)"; \
	else \
		echo "[DATAGEN CACHE] Miss ($(APP_NAME)) — running sbt"; \
		echo "  Scala $(GENERATOR_CLASS) $(GENERATOR_ARGS)"; \
		cd $(CHISEL_SSM) && sbt "test:runMain $(GENERATOR_CLASS) $(GENERATOR_ARGS)"; \
		mkdir -p "$(DATAGEN_CACHE_DIR)"; \
		tar cf "$$CACHED" -C "$(SBT_GEN_DIR)" .; \
		echo "[DATAGEN CACHE] Stored $(APP_NAME) → $$CACHE_KEY"; \
	fi
	@echo "Generating data.h from $(DATA_CFG)"
	@$(DATAGEN_PY) --swcfg $(DATA_CFG) > $@

endif
