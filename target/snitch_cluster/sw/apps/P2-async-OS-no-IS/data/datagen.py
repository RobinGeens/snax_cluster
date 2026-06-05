#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

import os
import sys
import pathlib
import importlib.util

import hjson

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "../../main/data"))

from datagen_base import DataGeneratorBase  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]

# Load main-tiled-oscore's generator by file path (the local file is also named datagen.py, so a
# plain `from datagen import ...` would re-import THIS module -> circular import).
_spec = importlib.util.spec_from_file_location(
    "_oscore_datagen", os.path.abspath(os.path.join(_HERE, "../../main-tiled-oscore/data/datagen.py"))
)
_oscore_datagen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oscore_datagen)


class DataGenerator(_oscore_datagen.DataGenerator):
    APP_NAME = "P2-async-OS-no-IS"

    def __init__(self, **kwargs):
        DataGeneratorBase.__init__(self, self.APP_NAME, **kwargs)
        self.phase1_scalars = {}
        self.phase2_scalars = {}
        local = pathlib.Path(__file__).resolve().parent / "params_in.hjson"
        for key, value in hjson.loads(local.read_text()).items():
            self.kwargs.setdefault(key, value)

    def run(self):
        # Both phases are emitted (the shared helper.c references M1_* and M2_*); this app only
        # consumes the M2_* Phase-2 constants, the async oscore_in ring scalars, and M33.
        self.save_params()
        self.check_tiling_constraints()
        self.build_Phase1_data()
        self.build_Phase2_data()


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
