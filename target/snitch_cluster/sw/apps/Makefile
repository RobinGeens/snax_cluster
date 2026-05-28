# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

SUBDIRS  = nop
SUBDIRS += main
SUBDIRS += vmamba
SUBDIRS += main-tiled-v3
SUBDIRS += vmamba-tiled
SUBDIRS += main-tiled
SUBDIRS += main-tiled-v2
SUBDIRS += fft
SUBDIRS += rmsnorm
SUBDIRS += fft-tiled
SUBDIRS += einfft-tiled
# SUBDIRS += fft-3way
# SUBDIRS += osgemm
# SUBDIRS += isgemm
# SUBDIRS += simd
# SUBDIRS += osgemm-tiled
# SUBDIRS += isgemm-tiled
# SUBDIRS += is-osgemm-tiled
# SUBDIRS += fft-3way-tiled
# SUBDIRS += main-full
# SUBDIRS += einfft
# SUBDIRS += streamer-burner
# SUBDIRS += core-burner
# SUBDIRS += suc-only


# Apps that have a data generator (auto-detected by presence of data/Makefile)
DATAGEN_APPS := $(foreach app,$(SUBDIRS),$(if $(wildcard $(app)/data/Makefile),$(app)))

.PHONY: all clean list-apps clean-cache $(SUBDIRS)

all: $(SUBDIRS)

list-apps:
	@echo $(SUBDIRS)

clean-cache:
	@$(foreach app,$(DATAGEN_APPS),$(MAKE) --no-print-directory -C $(app)/data clean-cache;)

clean:

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)
