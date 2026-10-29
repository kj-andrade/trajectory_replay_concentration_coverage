"""Configuration loading.

Every parameter the experiments read lives in a YAML file, and the resolved
dictionary is snapshotted next to the results, so a run can be reproduced
from its own artefacts without reference to the code that produced it.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

DEFAULTS = {
    "base_seed": 20260922,
    "seeds": [0, 1, 2, 3, 4],
    "n_queries": 500,
    "flow_time": 0.5,
    "out": "results",
    "cache": "results/cache",
    "save_per_query": True,
    "tol": 0.10,
}


def load(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg.update(loaded)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["out"] = str(cfg["out"])
    cfg["cache"] = str(cfg["cache"])
    return cfg


def geomspace(spec):
    """A log-spaced grid written in YAML as {lo, hi, n}."""
    return list(np.geomspace(float(spec["lo"]), float(spec["hi"]),
                             int(spec["n"])))


def exp_cfg(cfg, name):
    """The block for one experiment, with the global defaults folded in."""
    block = dict(cfg.get(name, {}) or {})
    for k in ("seeds", "n_queries", "flow_time", "tol", "save_per_query"):
        block.setdefault(k, cfg[k])
    return block
