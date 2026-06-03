# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@esat.kuleuven.be>

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR ?= $(realpath $(MK_DIR))
CURR_DIR ?= $(DATA_DIR)

# App-specific. Reuses the generic ISGEMM Scala golden-data generator (keyed by name=).
APP_NAME = isgemm-tiled-async
GENERATOR_CLASS = datagen.DataGeneratorISGEMM

include $(MK_DIR)/../../common-datagen.mk
