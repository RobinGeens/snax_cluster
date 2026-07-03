# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>
# Modified by: Robin Geens <robin.geens@kuleuven.be>

# Autodiscover apps: every subdirectory containing a Makefile is an app.
SUBDIRS := $(sort $(patsubst %/,%,$(dir $(wildcard */Makefile))))

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
