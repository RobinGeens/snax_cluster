# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

SUBDIRS  = lto
SUBDIRS += nop
# Tests below don't work with the generic runtime
ifneq ($(filter $(SELECT_RUNTIME),rtl banshee),)
SUBDIRS += blas/axpy
SUBDIRS += blas/gemm
SUBDIRS += dnn/batchnorm
SUBDIRS += dnn/conv2d
SUBDIRS += dnn/fusedconv
SUBDIRS += dnn/gelu
SUBDIRS += dnn/gemm
SUBDIRS += dnn/layernorm
SUBDIRS += dnn/linear
SUBDIRS += dnn/maxpool
SUBDIRS += dnn/softmax
endif

# SimbaCore tests and programs
ifeq ($(CFG_OVERRIDE), cfg/snax_simbacore_cluster.hjson)
SUBDIRS += main
SUBDIRS += osgemm
SUBDIRS += osgemm-tiled
SUBDIRS += isgemm
SUBDIRS += isgemm-tiled
SUBDIRS += simd
SUBDIRS += fft
SUBDIRS += fft-tiled
SUBDIRS += rmsnorm
SUBDIRS += main-full
SUBDIRS += main-tiled
SUBDIRS += streamer-burner
SUBDIRS += core-burner
SUBDIRS += suc
endif

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)
