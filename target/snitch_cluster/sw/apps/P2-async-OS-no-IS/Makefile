# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Robin Geens <robin.geens@esat.kuleuven.be>

APP = P2-async-OS-no-IS

# Reuse main-tiled-oscore's generated data (M2_* + async oscore_in ring scalars/bounds + M33 mode).
INCDIRS = ../main-tiled-oscore/data \
          ../../snax/simbacore/include \
          ../main/src

# SimbaCore support library
SIMBACORE_LIB_DIR := $(abspath ../../snax/simbacore)
SIMBACORE_LIB_OBJ := $(SIMBACORE_LIB_DIR)/build/snax-simbacore-lib.o

# Include this binary in the final build
RISCV_LDFLAGS += $(SIMBACORE_LIB_OBJ)

SRCS = src/main.c

include ../common.mk
include ../main-tiled-oscore/data/Makefile

$(ELF): $(SIMBACORE_LIB_OBJ)

$(SIMBACORE_LIB_OBJ):
	$(MAKE) -C $(SIMBACORE_LIB_DIR) $(SIMBACORE_LIB_OBJ)

$(DEP): $(DATA_H)
