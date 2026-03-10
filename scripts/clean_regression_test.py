#!/usr/bin/env python3
"""
Clean up regression test output folders that are not successful builds.
This is very hardcoded. Changing the summary will break this script.
"""
import re
import shutil
from pathlib import Path

MIN_TESTS_REPORTED = 4


def count_tests_reported(text: str) -> int:
    """Count test result lines; supports old (name: errors=N) and new table format."""
    if "Test Name" in text and "SimbaCore Cycles" in text:
        # New table format: rows like "nop    0   N/A   N/A" or "name  errors  simbacore  total"
        table_row = re.compile(r"^\S+\s+\d+\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s*$", re.MULTILINE)
        return len(table_row.findall(text))
    # Old format: "  nop: errors=0" etc.
    return sum(1 for line in text.splitlines() if ": errors=" in line)


base = Path(__file__).parent.parent / "regression_test_out"
removed = 0
for folder in base.iterdir():
    if folder.is_dir():
        summary = folder / "summary.log"
        if not summary.exists():
            shutil.rmtree(folder)
            removed += 1
            continue
        text = summary.read_text()
        if "Build: ✅ SUCCESS" not in text:
            shutil.rmtree(folder)
            removed += 1
        elif count_tests_reported(text) < MIN_TESTS_REPORTED:
            shutil.rmtree(folder)
            removed += 1
print(f"Removed {removed} folders.")
