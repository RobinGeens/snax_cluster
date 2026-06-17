#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>
#
# SUC-tiled datagen: reuses main-tiled's P2 generation (the M2_* tiled constants). The app
# only uses the SU-core subset (dt/BC, A, D, x, z, y) plus golden z (= M2_oscore_expected) and
# golden y (= M2_suc_expected); the extra osCore/IS-core M2_* constants are harmless.

import os
import sys
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
    APP_NAME = "SUC-tiled"

    def __init__(self, **kwargs):
        DataGeneratorBase.__init__(self, self.APP_NAME, **kwargs)
        self.phase1_scalars = {}
        self.phase2_scalars = {}
        local = self.params_in_path(__file__)
        for key, value in hjson.loads(local.read_text()).items():
            self.kwargs.setdefault(key, value)

    def run(self):
        # Both phases are emitted (the shared helper.c references M1_* and M2_*); SUC-tiled's
        # main.c only consumes the M2_* SU-core subset.
        self.save_params()
        self.check_tiling_constraints()
        self.build_Phase1_data()
        self.build_Phase2_data()
        self._run_memory_model()

    def _run_memory_model(self):
        app_dir = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("memory_model_suc_tiled",
                                                      os.path.join(app_dir, "memory_model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from memory_model_base import run_model_from_datagen  # type: ignore[import]

        self.lines_params.append(run_model_from_datagen(mod.build_report, app_dir))


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
