# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

SUBDIRS  = nop
SUBDIRS += main
# SUBDIRS += osgemm
# SUBDIRS += osgemm-tiled
# SUBDIRS += isgemm
# SUBDIRS += isgemm-tiled
# SUBDIRS += simd
SUBDIRS += fft
# SUBDIRS += fft-tiled
# SUBDIRS += fft-3way
# SUBDIRS += fft-3way-tiled
SUBDIRS += rmsnorm
# SUBDIRS += main-full
SUBDIRS += main-tiled
# SUBDIRS += einfft
# SUBDIRS += einfft-tiled
# SUBDIRS += streamer-burner
# SUBDIRS += core-burner
# SUBDIRS += suc-only
SUBDIRS += vmamba



.PHONY: all clean list-apps $(SUBDIRS)

all: $(SUBDIRS)

list-apps:
	@echo $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)
