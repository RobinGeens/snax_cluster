# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@esat.kuleuven.be>

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR ?= $(realpath $(MK_DIR))
CURR_DIR ?= $(DATA_DIR)

# App-specific
APP_NAME = vmamba-tiled
GENERATOR_CLASS = datagen.DataGeneratorVMamba

include $(MK_DIR)/../../common-datagen.mk
