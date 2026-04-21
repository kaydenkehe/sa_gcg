"""Pytest fixtures shared across the (CPU-only, dependency-light) test
suite. Tests that require GPU or a heavy dependency are marked ``slow``
or ``api`` and skipped by default.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_collection_modifyitems(config, items):
    skip_slow = "not slow" in (config.getoption("-k") or "")
    for item in items:
        if "slow" in item.keywords and not config.getoption("-k"):
            # default: don't skip; user can explicitly --select.
            continue
