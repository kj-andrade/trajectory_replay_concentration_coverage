"""Render Figs 5-7 from results/metrics_all.csv.

    python make_figures.py

Renders from whatever rows exist, so it is safe to call while the grid runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fc import load_config
from fc.figures import make_all, load


def main():
    cfg = load_config()
    rows = load()
    print(f"{len(rows)} metric rows")
    make_all(rows, off_name=cfg["off_family"])


if __name__ == "__main__":
    main()
