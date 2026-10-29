"""Finite-capacity experiment: do trained networks follow the closed-form
minimiser and inherit its replay behaviour? (Section 6.4 of the paper.)

Builds on ``fm_replay`` and imports its components rather than copying them.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from . import paths


def load_config(path=None):
    path = Path(path or paths.CONFIG)
    return yaml.safe_load(path.read_text(encoding="utf-8"))
