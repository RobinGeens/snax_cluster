#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Not released under license. All rights reserved.
#
# Author: Robin Geens <robin.geens@kuleuven.be>

import sys
import os
import importlib.util
import hjson

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
# Path in Occamy
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../main/data"))

# Dynamically load DataGenerator from the other datagen.py to avoid name clash
_this_dir = os.path.dirname(__file__)
_other_datagen_path = os.path.abspath(os.path.join(_this_dir, "../../main/data/datagen.py"))
_spec = importlib.util.spec_from_file_location("main_datagen", _other_datagen_path)
_main_datagen = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_main_datagen)


class DataGenerator(_main_datagen.DataGenerator):
    """Reuses main DataGenerator but overrides APP_NAME so data from generated/data/main-full will be
    used."""

    APP_NAME = "main-full"

    def __init__(self, **kwargs):
        # Make sure the params_in.hjson file of this app is used
        local = os.path.join(os.path.dirname(__file__), "params_in.hjson")
        with open(local) as f:
            for key, value in hjson.loads(f.read()).items():
                kwargs.setdefault(key, value)
        super().__init__(**kwargs)


from datagen_cli import main as datagen_cli_main  # type: ignore[import]


if __name__ == "__main__":
    datagen_cli_main(DataGenerator)
