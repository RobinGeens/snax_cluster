# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR ?= $(realpath $(MK_DIR))
CURR_DIR ?= $(DATA_DIR)

APP_NAME = p2-carry
GENERATOR_CLASS = datagen.DataGeneratorMain

include $(MK_DIR)/../../common-datagen.mk
