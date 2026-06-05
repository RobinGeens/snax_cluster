#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# P1-tiled-D datagen: reuses main-tiled's Phase-1 generation but emits ONLY the M1_* tiled
# constants (Phase 2 is not built). dInner tiling and BC bank-padding are inherited unchanged.

import os
import sys
import pathlib
import importlib.util

import hjson

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "../../main/data"))

from datagen_base import DataGeneratorBase  # type: ignore[import]
from datagen_cli import main as datagen_cli_main  # type: ignore[import]

# Load main-tiled's generator by file path (the local file is also named datagen.py, so a plain
# `from datagen import ...` would re-import THIS module -> circular import).
_spec = importlib.util.spec_from_file_location(
    "_main_tiled_datagen", os.path.abspath(os.path.join(_HERE, "../../main-tiled/data/datagen.py"))
)
_main_tiled_datagen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main_tiled_datagen)


class DataGenerator(_main_tiled_datagen.DataGenerator):
    APP_NAME = "P1-tiled-D"

    def __init__(self, **kwargs):
        # Bypass main-tiled's __init__ (its __file__ points at main-tiled/data); read THIS app's
        # params_in.hjson for the params scala does not propagate (nb_tiles, bc_pad_banks).
        DataGeneratorBase.__init__(self, self.APP_NAME, **kwargs)
        self.phase1_scalars = {}
        self.phase2_scalars = {}
        local = pathlib.Path(__file__).resolve().parent / "params_in.hjson"
        for key, value in hjson.loads(local.read_text()).items():
            self.kwargs.setdefault(key, value)

    def run(self):
        # Both phases are emitted (the shared helper.c references M1_* and M2_*); P1-tiled-D's
        # main.c only consumes the M1_* Phase-1 constants.
        self.save_params()
        self.check_tiling_constraints()
        self.build_Phase1_data()
        self.build_Phase2_data()


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
