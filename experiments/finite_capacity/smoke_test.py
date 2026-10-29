"""Smoke test: a short training run to check the pipeline before the grid.

lambda = 8, W = 512, a temporary bank of M = 200, 5,000 steps.  Passes if the
training loss decreases, delta_L at 5k is below delta_L at 500, and no NaN
appears.  Reports the trajectory of the fidelity r.

The M = 200 bank is the first 200 rows of the M = 2000 bank: the
bank construction truncates a single permutation, so the small bank is
nested inside the large one.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from fc import load_config
from fc.closed_form import closed_form, star_loss_batch
from fc.model import make_model
from fc.problem import build
from fc.train import lr_at

LAM, WIDTH, M_SMALL, STEPS = 8.0, 512, 200, 5000
CKPT = (500, 1000, 2000, 3000, 5000)
N_DL = 16384


def main():
    cfg = load_config()
    P, scal = build(cfg, verbose=False)
    d = int(scal["d"])
    u_rms = scal["u_rms"]

    C, U = P["C"][:M_SMALL], P["U"][:M_SMALL]
    mu0, mu1 = C, C + U
    V = float(2.0 * np.trace(np.cov(0.5 * (mu0 + mu1), rowvar=False,
                                    ddof=1)) / d)
    sigma2 = d * V / (LAM * math.log(M_SMALL))
    sigma = math.sqrt(sigma2)
    print(f"smoke test: M={M_SMALL} V={V:.4f} sigma^2={sigma2:.4f} "
          f"kappa_hat={d * V / sigma2:.2f}  (2 log M = "
          f"{2 * math.log(M_SMALL):.2f})")

    # Fixed evaluation queries from the small bank, and the v* they target.
    r = np.random.default_rng(20260923)
    i_gen = r.integers(0, M_SMALL, 500)
    eps = r.standard_normal((500, d))
    t_eval = 0.5
    x = (1 - t_eval) * mu0[i_gen] + t_eval * mu1[i_gen] + sigma * eps
    v_star, _, _ = closed_form(x, t_eval, mu0, mu1, U, sigma2)

    jd = r.integers(0, M_SMALL, N_DL)
    td = r.random(N_DL)
    epsd = r.standard_normal((N_DL, d))
    L_star, per_star = star_loss_batch(mu0, mu1, U, sigma2, jd, td, epsd)
    xd = mu0[jd] + td[:, None] * (mu1 - mu0)[jd] + sigma * epsd
    print(f"         L(v*) = {L_star:.6f}")

    torch.set_num_threads(8)
    net = make_model(d, WIDTH, seed=0)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr_peak"])
    g = torch.Generator().manual_seed(20260923)
    X0 = torch.as_tensor(mu0, dtype=torch.float32)
    X1 = torch.as_tensor(mu1, dtype=torch.float32)
    Ut = torch.as_tensor(U, dtype=torch.float32)
    B = int(cfg["batch"])
    sub = dict(cfg, steps=STEPS)

    print(f"\n{'step':>6} {'loss':>10} {'delta_L':>12} {'median r':>10}")
    hist, acc, n = [], 0.0, 0
    for step in range(STEPS):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, sub)
        j = torch.randint(0, M_SMALL, (B,), generator=g)
        t = torch.rand(B, generator=g)
        e = torch.randn(B, d, generator=g)
        xb = (1 - t)[:, None] * X0[j] + t[:, None] * X1[j] + sigma * e
        loss = ((net(xb, t) - Ut[j]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step()
        acc += float(loss.detach()); n += 1

        if step + 1 in CKPT:
            net.eval()
            with torch.no_grad():
                v_th = net(torch.as_tensor(x, dtype=torch.float32),
                           torch.full((500,), t_eval)).double().numpy()
                vd = net(torch.as_tensor(xd, dtype=torch.float32),
                         torch.as_tensor(td, dtype=torch.float32)
                         ).double().numpy()
            net.train()
            rr = np.median(np.linalg.norm(v_th - v_star, axis=1) / u_rms)
            dL = float((((vd - U[jd]) ** 2).mean(1) - per_star).mean())
            hist.append((step + 1, acc / n, dL, float(rr)))
            print(f"{step + 1:6d} {acc / n:10.5f} {dL:+12.4e} {rr:10.4f}")
            acc, n = 0.0, 0

    losses = [h[1] for h in hist]
    dls = [h[2] for h in hist]
    ok_loss = losses[-1] < losses[0]
    ok_dl = dls[-1] < dls[0]
    ok_fin = all(np.isfinite([*losses, *dls, *[h[3] for h in hist]]))
    print(f"\n  loss decreased      : {ok_loss}  ({losses[0]:.5f} -> {losses[-1]:.5f})")
    print(f"  delta_L(5k)<delta_L(500): {ok_dl}  ({dls[0]:.4e} -> {dls[-1]:.4e})")
    print(f"  all finite          : {ok_fin}")
    print(f"  r trajectory        : "
          + ", ".join(f"{h[0]}:{h[3]:.3f}" for h in hist))
    passed = ok_loss and ok_dl and ok_fin
    print(f"\n{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
