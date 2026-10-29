"""Run one experiment and write its artefacts.

    python -m scripts.run --exp onset      --config configs/final.yaml
    python -m scripts.run --exp coverage   --config configs/final.yaml
    python -m scripts.run --exp anisotropy --config configs/final.yaml

Seeds are derived from the spawn key, so results do not depend on execution
order and --systems may be used to split a run across processes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from fm_replay import config as C
from fm_replay.results import (write_csv, write_per_query, write_env,
                               write_config, Timer)

EXPERIMENTS = ("onset", "coverage", "anisotropy")


def _quick(name, block):
    """A smoke-test shrink of a grid. Not publication numbers."""
    block = dict(block)
    block["seeds"] = block["seeds"][:2]
    block["n_queries"] = 100
    if name == "onset":
        block.update(dims_l96=[8, 32], dims_gauss=[8, 32], k_orders=[1, 2],
                     M_main=300, M_onset=[30, 300, 3000],
                     onset_systems=["Gauss-32", "L96-32-onset"],
                     flow_times=[0.5, 0.35],
                     kappa_grid={"lo": 0.1, "hi": 1000.0, "n": 14})
    elif name == "coverage":
        block.update(grid=[["L63", [200, 2000]], ["L96-16", [2000]]],
                     kappa_grid={"lo": 0.3, "hi_mult": 64, "n": 8})
    elif name == "anisotropy":
        block.update(systems=["L96-64", "L96-64-F4-bump"], M=2000,
                     kappa_grid={"lo": 1.0, "hi_mult": 16, "n": 6})
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=EXPERIMENTS + ("all",))
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test only; NOT publication numbers")
    a = ap.parse_args()

    cfg = C.load(a.config, overrides=dict(out=a.out))
    todo = EXPERIMENTS if a.exp == "all" else (a.exp,)

    for name in todo:
        block = C.exp_cfg(cfg, name)
        if a.quick:
            block = _quick(name, block)

        out = Path(cfg["out"]) / name
        cache = Path(cfg["cache"])
        timer = Timer()
        timer.start(name)

        print(f"=== {name} ===", flush=True)
        per_query = {}
        if name == "onset":
            from fm_replay.experiments.onset import run
            rows = run(block, cache)
        elif name == "coverage":
            from fm_replay.experiments.coverage import run
            rows, per_query = run(block, cache)
        else:
            from fm_replay.experiments.anisotropy import run
            rows = run(block, cache)
        timer.stop()

        write_csv(rows, out / "rows.csv")
        if per_query and block.get("save_per_query"):
            write_per_query(per_query, out / "per_query.npz")
        write_config(dict(experiment=name, quick=bool(a.quick),
                          base_seed=cfg["base_seed"], resolved=block),
                     out / "config.yaml")
        write_env(out / "env.txt")
        timer.write(out / "timing.json")
        print(f"  {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
