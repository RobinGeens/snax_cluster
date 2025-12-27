#!/usr/bin/env python3
import shutil
from pathlib import Path

base = Path(__file__).parent.parent / "regression_test_out"
for folder in base.iterdir():
    if folder.is_dir():
        summary = folder / "summary.log"
        if summary.exists() and "Build: ❌ FAILED" in summary.read_text():
            shutil.rmtree(folder)
