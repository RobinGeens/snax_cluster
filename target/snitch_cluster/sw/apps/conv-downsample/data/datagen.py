#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

# conv-downsample runs a 3x3 stride-2 Conv2d as an im2col, K-tiled IS-GEMM. The device
# kernel and streamer programming are identical to isgemm-tiled; only the golden data
# differs (DataGeneratorConvDownsample.scala).
# See docs/dataflow/21_conv_downsample.md.

import os
import sys
import importlib.util

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
# Path in Occamy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))

# Dynamically load DataGenerator from isgemm-tiled's datagen.py to avoid the name clash.
_this_dir = os.path.dirname(__file__)
_other_datagen_path = os.path.abspath(os.path.join(_this_dir, "../../isgemm-tiled/data/datagen.py"))
_spec = importlib.util.spec_from_file_location("isgemm_tiled_datagen", _other_datagen_path)
_isgemm_datagen = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_isgemm_datagen)


class DataGenerator(_isgemm_datagen.DataGenerator):
    """Reuses isgemm-tiled's DataGenerator (identical streamer programming); overrides
    APP_NAME so the conv-downsample golden data in generated/data/conv-downsample is used."""

    APP_NAME = "conv-downsample"


from datagen_cli import main as datagen_cli_main  # type: ignore[import]  # noqa: E402


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
