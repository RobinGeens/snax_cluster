# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

SUBDIRS  = nop
SUBDIRS += main
SUBDIRS += vmamba
SUBDIRS += fft
# SUBDIRS += fft-3way
# SUBDIRS += osgemm
SUBDIRS += rmsnorm
# SUBDIRS += isgemm
# SUBDIRS += simd
# SUBDIRS += osgemm-tiled
# SUBDIRS += isgemm-tiled
# SUBDIRS += fft-tiled
# SUBDIRS += fft-3way-tiled
# SUBDIRS += main-full
# SUBDIRS += einfft
SUBDIRS += main-tiled
SUBDIRS += einfft-tiled
# SUBDIRS += streamer-burner
# SUBDIRS += core-burner
# SUBDIRS += suc-only


CHISEL_SSM := $(shell bender path chisel-ssm)
DATAGEN_MANIFEST := $(abspath .datagen_manifest)
DATAGEN_STAMP := $(abspath .datagen_stamp)
export DATAGEN_MANIFEST

# Apps that have a data generator (auto-detected by presence of data/Makefile)
DATAGEN_APPS := $(foreach app,$(SUBDIRS),$(if $(wildcard $(app)/data/Makefile),$(app)))
DATAGEN_INPUTS := $(foreach app,$(DATAGEN_APPS),$(app)/data/params_in.hjson)

.PHONY: all clean list-apps datagen-all $(SUBDIRS)

all: $(DATAGEN_STAMP) $(SUBDIRS)

list-apps:
	@echo $(SUBDIRS)

# Stamp-based bulk datagen: only re-runs sbt when a params_in.hjson changes
$(DATAGEN_STAMP): $(DATAGEN_INPUTS)
	@rm -f $(DATAGEN_MANIFEST)
	@$(foreach app,$(DATAGEN_APPS),$(MAKE) -C $(app) datagen-register;)
	@echo "Running all Scala data generators in a single sbt session..."
	cd $(CHISEL_SSM) && sbt "$$(paste -sd ';' $(DATAGEN_MANIFEST))"
	@rm -f $(DATAGEN_MANIFEST)
	@touch $@
	@echo "Bulk datagen complete."

# Force re-run of all generators regardless of timestamps
datagen-all:
	@rm -f $(DATAGEN_STAMP)
	@$(MAKE) $(DATAGEN_STAMP)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)
