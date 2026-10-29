"""Run artefacts.

Each experiment writes results/<exp>/ containing rows.csv, optionally
per_query.npz, plus the three files that make a run auditable after the
fact: the resolved config, the environment, and the timings.
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Row tables
# ----------------------------------------------------------------------

def write_csv(rows, path):
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return path


def _coerce(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def read_csv(path):
    """Read rows back with numbers restored, so plotters see floats."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [{k: _coerce(v) for k, v in r.items()} for r in csv.DictReader(f)]


def write_per_query(arrays, path):
    """Raw per-query metrics, keyed by a flattened cell name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "not a git repository"
    except Exception:
        return "unavailable"


def _pip_freeze():
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=120)
        return out.stdout
    except Exception:
        return "unavailable\n"


def write_env(path):
    import os
    import scipy
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = [
        f"python          {sys.version.splitlines()[0]}",
        f"numpy           {np.__version__}",
        f"scipy           {scipy.__version__}",
        f"platform        {platform.platform()}",
        f"processor       {platform.processor()}",
        f"cpu_count       {os.cpu_count()}",
        f"OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS', 'unset')}",
        f"MKL_NUM_THREADS {os.environ.get('MKL_NUM_THREADS', 'unset')}",
        f"git_commit      {_git_commit()}",
        "",
        "--- pip freeze ---",
    ]
    path.write_text("\n".join(head) + "\n" + _pip_freeze(), encoding="utf-8")
    return path


def write_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        text = yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)
    except Exception:
        text = json.dumps(cfg, indent=2, sort_keys=True, default=str)
    path.write_text(text, encoding="utf-8")
    return path


class Timer:
    """Wall-clock time per system, written to timing.json."""

    def __init__(self):
        self.times = {}
        self._t0 = None
        self._key = None

    def start(self, key):
        self._key, self._t0 = key, time.time()

    def stop(self):
        if self._key is not None:
            self.times[self._key] = self.times.get(self._key, 0.0) + (
                time.time() - self._t0)
            self._key = self._t0 = None

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(per_system=self.times,
                       total_seconds=float(sum(self.times.values())))
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
