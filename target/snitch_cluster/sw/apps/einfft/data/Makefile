# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@esat.kuleuven.be>

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR ?= $(realpath $(MK_DIR))
CURR_DIR ?= $(DATA_DIR)

# App-specific
APP_NAME = einfft
GENERATOR_CLASS = datagen.DataGeneratorComplexMlp

include $(MK_DIR)/../../common-datagen.mk

# OS-core flow (M3_* references in flattenA per-branch).
# Must come AFTER common-datagen.mk because that include uses ':=' on
# GENERATOR_ARGS, which would otherwise clobber any prior `+= core=os`.
GENERATOR_ARGS += core=os
