# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@esat.kuleuven.be>

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR ?= $(realpath $(MK_DIR))
CURR_DIR ?= $(DATA_DIR)

# Reuse the einfft-tiled-is-osgemm golden data (identical arithmetic); only the on-chip
# layout differs. APP_NAME selects the shared sbt output dir + datagen cache.
APP_NAME = einfft-tiled-is-osgemm
GENERATOR_CLASS = datagen.DataGeneratorEinfftMlpIsOs

include $(MK_DIR)/../../common-datagen.mk
