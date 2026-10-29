"""One training run, plus the evaluation at checkpoints.

One set of hyperparameters serves every cell: no per-cell tuning, no early
stopping, no EMA, no coupling and no augmentation.  The training draw is
exactly the empirical flow-matching objective on the same bank, path and sigma as
the v* it is compared against.

Runs are idempotent.  A run directory whose metrics.csv already contains the
final checkpoint is skipped, so the grid can be resumed after an
interruption without recomputing anything.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import evaluate
from .model import make_model

METRIC_FIELDS = ["run_id", "d", "M", "lambda", "sigma2", "kappa_hat_t", "W",
                 "seed", "step", "t", "set", "metric", "median", "q25", "q75"]


def run_id(d, M, lam, width, seed):
    return f"d{d}_M{M}_lam{lam:g}_W{width}_s{seed}"


def lr_at(step, cfg):
    """500-step linear warmup to lr_peak, then cosine decay to lr_final."""
    warm, total = int(cfg["warmup"]), int(cfg["steps"])
    peak, final = float(cfg["lr_peak"]), float(cfg["lr_final"])
    if step < warm:
        return peak * (step + 1) / warm
    p = (step - warm) / max(total - warm, 1)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * min(p, 1.0)))


def _is_complete(path, final_step):
    if not path.exists():
        return False
    try:
        with path.open(newline="") as f:
            return any(int(float(r["step"])) == final_step
                       for r in csv.DictReader(f))
    except Exception:
        return False


def train_run(cfg, P, scal, lam, width, seed, outdir, device="cpu",
              threads=None, verbose=True):
    if threads:
        torch.set_num_threads(int(threads))

    d, M = int(scal["d"]), int(scal["M"])
    rid = run_id(d, M, lam, width, seed)
    outdir = Path(outdir) / rid
    outdir.mkdir(parents=True, exist_ok=True)
    mpath = outdir / "metrics.csv"
    total = int(cfg["steps"])
    ckpts = sorted(int(c) for c in cfg["checkpoints"])

    if _is_complete(mpath, ckpts[-1]):
        if verbose:
            print(f"  [skip] {rid}", flush=True)
        return mpath

    li = [float(x) for x in cfg["lambdas"]].index(float(lam))
    sigma2 = scal["sigma2"][str(float(lam))]
    sigma = math.sqrt(sigma2)
    B = int(cfg["batch"])
    clip = float(cfg["grad_clip"])

    X0 = torch.as_tensor(P["C"], dtype=torch.float32, device=device)
    U = torch.as_tensor(P["U"], dtype=torch.float32, device=device)
    X1 = X0 + U

    net = make_model(d, width, seed, device)
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["lr_peak"]),
                           betas=(0.9, 0.999), weight_decay=0.0)
    # The training RNG is seeded by the run seed and is separate from every
    # evaluation RNG.
    g = torch.Generator(device="cpu").manual_seed(20260923 + 7919 * int(seed))

    (outdir / "config.json").write_text(json.dumps(dict(
        run_id=rid, d=d, M=M, **{"lambda": float(lam)}, sigma2=sigma2,
        W=int(width), seed=int(seed), steps=total, batch=B,
        checkpoints=ckpts, lr_peak=cfg["lr_peak"], lr_final=cfg["lr_final"],
        warmup=cfg["warmup"], grad_clip=clip, device=str(device),
        n_params=net.n_params), indent=2), encoding="utf-8")

    with mpath.open("w", newline="") as fm, (outdir / "log.csv").open(
            "w", newline="") as fl:
        wm = csv.DictWriter(fm, fieldnames=METRIC_FIELDS)
        wm.writeheader()
        wl = csv.DictWriter(fl, fieldnames=["step", "seconds_per_step",
                                            "loss", "delta_L"])
        wl.writeheader()

        run_loss, n_loss, t0 = 0.0, 0, time.perf_counter()
        for step in range(total):
            for grp in opt.param_groups:
                grp["lr"] = lr_at(step, cfg)
            j = torch.randint(0, M, (B,), generator=g).to(device)
            t = torch.rand(B, generator=g).to(device)
            eps = torch.randn(B, d, generator=g).to(device)
            x = (1 - t)[:, None] * X0[j] + t[:, None] * X1[j] + sigma * eps
            loss = ((net(x, t) - U[j]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            run_loss += float(loss.detach()); n_loss += 1
            done = step + 1
            if done % 500 == 0 and done not in ckpts:
                sps = (time.perf_counter() - t0) / n_loss
                wl.writerow(dict(step=done, seconds_per_step=f"{sps:.6f}",
                                 loss=f"{run_loss / n_loss:.6f}", delta_L=""))
                fl.flush()
                run_loss, n_loss, t0 = 0.0, 0, time.perf_counter()

            if done in ckpts:
                sps = (time.perf_counter() - t0) / max(n_loss, 1)
                rows, arrays, dL, se = evaluate(
                    net, P, scal, cfg, li, float(lam), int(width), int(seed),
                    done, rid, device)
                for r in rows:
                    wm.writerow({k: ("" if r.get(k) is None else r.get(k, ""))
                                 for k in METRIC_FIELDS})
                fm.flush()
                np.savez_compressed(outdir / f"queries_{done}.npz", **arrays)
                if done == ckpts[-1]:
                    # The final weights, so any further reference field can be
                    # measured later without retraining the grid.
                    torch.save({k: v.cpu() for k, v in
                                net.state_dict().items()},
                               outdir / "weights_final.pt")
                wl.writerow(dict(step=done, seconds_per_step=f"{sps:.6f}",
                                 loss=f"{run_loss / max(n_loss, 1):.6f}",
                                 delta_L=f"{dL:.6e}"))
                fl.flush()
                if verbose:
                    print(f"  [{rid}] step {done:6d}  loss "
                          f"{run_loss / max(n_loss, 1):.5f}  dL {dL:+.4e} "
                          f"(SE {se:.1e})  {sps * 1000:.1f} ms/step", flush=True)
                run_loss, n_loss, t0 = 0.0, 0, time.perf_counter()
    return mpath
