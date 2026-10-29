"""Run the full grid: time a few steps, launch every run, collect the table.

    python run_grid.py                 # time, launch, collect
    python run_grid.py --skip-timing   # resume without re-timing
    python run_grid.py --collect-only  # rebuild results/metrics_all.csv

Runs are ordered with seed outermost, so an interrupted grid still holds the
full lambda-by-width grid at fewer seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fc import load_config, paths
from fc.problem import build
from fc.train import METRIC_FIELDS, run_id

PY = sys.executable
HERE = Path(__file__).resolve().parent

TIMING_CHILD = r"""
import sys, time
sys.path.insert(0, r"{here}")
import torch
torch.set_num_threads({threads})
from fc.model import VelocityMLP
D, M, B = 64, {M}, {batch}
net = VelocityMLP(D, {width})
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
X0 = torch.randn(M, D); U = torch.randn(M, D)
g = torch.Generator().manual_seed(0)
def one():
    j = torch.randint(0, M, (B,), generator=g)
    t = torch.rand(B, generator=g)
    x = X0[j] + t[:, None] * U[j] + 0.1 * torch.randn(B, D, generator=g)
    loss = ((net(x, t) - U[j]) ** 2).mean()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
for _ in range(25): one()
t0 = time.perf_counter()
for _ in range({steps}): one()
print((time.perf_counter() - t0) / {steps})
"""


def _spawn(cmd, threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), PYTHONIOENCODING="utf-8")
    return subprocess.Popen(cmd, env=env, cwd=str(HERE))


def time_grid(cfg, steps=100):
    """Seconds per step, per-run ETA and total ETA for the grid."""
    threads = int(cfg["threads_per_proc"])
    nproc = int(cfg["concurrency"])
    env = dict(os.environ, OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    per_step = {}
    for width in cfg["widths"]:
        code = TIMING_CHILD.format(here=str(HERE), threads=threads,
                                   width=width, M=cfg["M"],
                                   batch=cfg["batch"], steps=steps)
        procs = [subprocess.Popen([PY, "-c", code], env=env,
                                  stdout=subprocess.PIPE, text=True)
                 for _ in range(nproc)]
        vals = [float(p.communicate()[0].strip()) for p in procs]
        per_step[int(width)] = sum(vals) / len(vals)

    print(f"\nTiming on {nproc} concurrent processes x {threads} "
          f"threads")
    print(f"{'W':>6} {'s/step':>10} {'run ETA (h)':>12} {'runs':>6} "
          f"{'block ETA (h)':>14}")
    total = 0.0
    n_seeds, n_lam = len(cfg["seeds"]), len(cfg["lambdas"])
    for width in cfg["widths"]:
        s = per_step[int(width)]
        runs = n_seeds * n_lam
        run_h = s * int(cfg["steps"]) / 3600
        block = run_h * runs / nproc
        total += block
        print(f"{width:6d} {s:10.5f} {run_h:12.3f} {runs:6d} {block:14.3f}")
    print(f"{'total':>6} {'':>10} {'':>12} {n_seeds * n_lam * len(cfg['widths']):6d} "
          f"{total:14.3f}")
    return total, per_step


def apply_fallbacks(cfg, eta, per_step):
    """Shrink the grid in a fixed order until it fits the time budget."""
    budget = float(cfg["TIME_BUDGET_HOURS"])
    applied = []

    def recompute():
        nproc = int(cfg["concurrency"])
        tot = 0.0
        for width in cfg["widths"]:
            runs = len(cfg["seeds"]) * len(cfg["lambdas"])
            tot += per_step[int(width)] * int(cfg["steps"]) * runs / 3600 / nproc
        return tot

    if eta <= budget:
        print(f"\nETA {eta:.2f} h is within the {budget:g} h budget; "
              f"launching the full grid.")
        return cfg, applied

    for name, change in (
            ("seeds 3 -> 2", lambda c: c.update(seeds=[0, 1])),
            ("50k -> 30k steps", lambda c: c.update(
                steps=30000, checkpoints=[1000, 3000, 10000, 30000])),
            ("lambda -> {1, 8, 32}", lambda c: c.update(lambdas=[1, 8, 32]))):
        change(cfg)
        applied.append(name)
        eta = recompute()
        print(f"  fallback: {name} -> ETA {eta:.2f} h")
        if eta <= budget:
            return cfg, applied

    raise SystemExit(f"ETA {eta:.2f} h still exceeds the {budget:g} h budget "
                     f"after every fallback; stopping.")


def cells(cfg):
    """Seed outermost, so a partial grid is still a full lambda-by-W grid."""
    out = []
    for seed in cfg["seeds"]:
        for width in cfg["widths"]:
            for lam in cfg["lambdas"]:
                out.append((float(lam), int(width), int(seed)))
    return out


def launch(cfg, todo, extra_args=()):
    nproc = int(cfg["concurrency"])
    threads = int(cfg["threads_per_proc"])
    pending = list(todo)
    running = []
    t0 = time.perf_counter()
    done = 0
    while pending or running:
        while pending and len(running) < nproc:
            lam, width, seed = pending.pop(0)
            cmd = [PY, "run_one.py", "--lam", f"{lam:g}", "--width",
                   str(width), "--seed", str(seed), "--threads", str(threads),
                   *extra_args]
            running.append(((lam, width, seed), _spawn(cmd, threads)))
        time.sleep(2.0)
        still = []
        for cell, p in running:
            if p.poll() is None:
                still.append((cell, p))
            else:
                done += 1
                mark = "ok" if p.returncode == 0 else f"FAILED ({p.returncode})"
                print(f"[{done}/{len(todo)}] lam={cell[0]:g} W={cell[1]} "
                      f"s={cell[2]} {mark}  "
                      f"[{(time.perf_counter() - t0) / 60:.1f} min elapsed]",
                      flush=True)
        running = still
    return time.perf_counter() - t0


def collect(cfg, out=None):
    out = Path(out or paths.RESULTS) / "metrics_all.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        w.writeheader()
        for path in sorted(paths.RUNS.glob("*/metrics.csv")):
            with path.open(newline="") as g:
                for r in csv.DictReader(g):
                    w.writerow({k: r.get(k, "") for k in METRIC_FIELDS})
                    n += 1
    print(f"collected {n} rows -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)

    if a.collect_only:
        collect(cfg)
        return

    print("=== building the fixed evaluation problem ===", flush=True)
    t0 = time.perf_counter()
    P, scal = build(cfg)
    print(f"  problem ready in {time.perf_counter() - t0:.1f} s  "
          f"(u_rms={scal['u_rms']:.4f}, vscale={scal['vscale']:.4f}, "
          f"s_u={scal['s_u']:.4f}, V={scal['V_half']:.4f}, "
          f"max simplex violation {scal['max_simplex_violation']:.2e})",
          flush=True)

    applied = []
    if not a.skip_timing:
        eta, per_step = time_grid(cfg)
        cfg, applied = apply_fallbacks(cfg, eta, per_step)

    todo = cells(cfg)
    extra = ["--steps", str(cfg["steps"])] if applied else []
    print(f"\n=== launching {len(todo)} runs ===", flush=True)
    wall = launch(cfg, todo, extra)
    print(f"\ngrid complete in {wall / 3600:.2f} h", flush=True)

    collect(cfg)
    (paths.RESULTS / "grid.json").write_text(json.dumps(
        dict(seeds=cfg["seeds"], lambdas=cfg["lambdas"], widths=cfg["widths"],
             steps=cfg["steps"], checkpoints=cfg["checkpoints"],
             fallbacks_applied=applied, wall_seconds=wall,
             concurrency=cfg["concurrency"],
             threads_per_proc=cfg["threads_per_proc"]), indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
