"""Render figures and tables from saved rows. Never simulates.

    python -m scripts.plot --all --results results --out figures
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from fm_replay import config as C
from fm_replay.results import read_csv
from fm_replay import tables

EXPERIMENTS = ("onset", "coverage", "anisotropy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--exp", choices=EXPERIMENTS, default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--config", default="configs/final.yaml")
    a = ap.parse_args()

    cfg = C.load(a.config)
    tol = float(cfg["tol"])
    res, out = Path(a.results), Path(a.out)
    todo = EXPERIMENTS if (a.all or a.exp is None) else (a.exp,)
    notes = []

    for name in todo:
        rows = read_csv(res / name / "rows.csv")
        if not rows:
            print(f"  {name}: rows.csv missing, skipped")
            continue

        if name == "onset":
            from fm_replay.figures.fig_onset import figure_onset
            paths = figure_onset(rows, out)
            notes.append(tables.table1(rows))
        elif name == "coverage":
            from fm_replay.figures.fig_coverage import figure_coverage
            paths = figure_coverage(rows, out)
            notes.append(tables.table2(rows))
        else:
            from fm_replay.figures.fig_anisotropy import figure_anisotropy
            paths = figure_anisotropy(rows, out, tol=tol)
            notes.append(tables.table3(rows))
            notes.append(tables.table4(rows, tol=tol))
        print(f"  {name}: {paths[1]}")

    if notes:
        out.mkdir(parents=True, exist_ok=True)
        p = out / "reported_numbers.md"
        p.write_text("\n\n---\n\n".join(notes), encoding="utf-8")
        print(f"  tables: {p}")


if __name__ == "__main__":
    main()
