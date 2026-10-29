"""Train and evaluate a single cell of the grid.

    python run_one.py --lam 8 --width 512 --seed 0

Called by ``run_grid.py`` as a subprocess so that thread counts can be set
per worker, and usable on its own to repair one cell.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from fc import load_config, paths
from fc.problem import build
from fc.train import train_run


def pick_device(cfg):
    want = str(cfg.get("DEVICE", "auto")).lower()
    if want == "cpu":
        return "cpu"
    if torch.cuda.is_available() and want in ("auto", "cuda"):
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.steps:
        cfg["steps"] = a.steps
        cfg["checkpoints"] = [c for c in cfg["checkpoints"] if c <= a.steps]
    P, scal = build(cfg, verbose=False)
    train_run(cfg, P, scal, a.lam, a.width, a.seed,
              a.out or paths.RUNS, device=pick_device(cfg),
              threads=a.threads or cfg.get("threads_per_proc"))


if __name__ == "__main__":
    main()
