# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Robin Geens <robin.geens@esat.kuleuven.be>

APP = suc-async

INCDIRS = data \
          ../../snax/simbacore/include \
          ../main/src

# SimbaCore support library
SIMBACORE_LIB_DIR := $(abspath ../../snax/simbacore)
SIMBACORE_LIB_OBJ := $(SIMBACORE_LIB_DIR)/build/snax-simbacore-lib.o

RISCV_LDFLAGS += $(SIMBACORE_LIB_OBJ)

SRCS = src/main.c

include ../common.mk
include ./data/Makefile

$(ELF): $(SIMBACORE_LIB_OBJ)

$(SIMBACORE_LIB_OBJ):
	$(MAKE) -C $(SIMBACORE_LIB_DIR) $(SIMBACORE_LIB_OBJ)

$(DEP): $(DATA_H)
