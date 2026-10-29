"""Locate the repository root and make ``fm_replay`` importable.

This experiment imports the closed-form components rather than copying them,
so every module here begins by putting the repository root on the path.
"""
from __future__ import annotations

import sys
from pathlib import Path

# <repo>/experiments/finite_capacity/fc/paths.py
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS = EXP / "runs"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
CACHE = EXP / "cache"
CONFIG = EXP / "config" / "grid.yaml"
