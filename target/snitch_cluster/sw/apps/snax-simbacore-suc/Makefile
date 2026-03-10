# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
# Robin Geens <robin.geens@esat.kuleuven.be>

APP = snax-simbacore-suc

INCDIRS = ../snax-simbacore-main-full/data \
          ../../snax/simbacore/include \
          ../snax-simbacore-main/src

# SimbaCore support library
SIMBACORE_LIB_DIR := $(abspath ../../snax/simbacore)
SIMBACORE_LIB_OBJ := $(SIMBACORE_LIB_DIR)/build/snax-simbacore-lib.o

# Include this binary in the final build
RISCV_LDFLAGS += $(SIMBACORE_LIB_OBJ)

SRCS = src/snax-simbacore-suc.c

include ../common.mk
include ../snax-simbacore-main-full/data/Makefile

$(ELF): $(SIMBACORE_LIB_OBJ)

$(SIMBACORE_LIB_OBJ):
	$(MAKE) -C $(SIMBACORE_LIB_DIR) $(SIMBACORE_LIB_OBJ)

$(DEP): $(DATA_H)
