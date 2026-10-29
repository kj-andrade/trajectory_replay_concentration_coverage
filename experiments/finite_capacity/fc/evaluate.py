"""Per-checkpoint evaluation.

The network is trained in float32; every quantity reported here is computed
in float64 on CPU against the cached closed-form arrays, so the comparison
between v_theta and v* never runs through a second float32 reduction.
"""
from __future__ import annotations

import numpy as np
import torch

from .closed_form import replay_indicator
from .problem import _key

SET_INDEX = {"IN": 0, "HO": 1, "OFF": 2}
INDICATOR = {"rep_star", "rep_theta", "escape", "escape_u"}


@torch.no_grad()
def field(net, x, t, device="cpu", chunk=8192):
    """v_theta(x, t) in float64, from a float32 network."""
    net.eval()
    out = np.empty_like(x)
    for s in range(0, len(x), chunk):
        e = min(s + chunk, len(x))
        xb = torch.as_tensor(x[s:e], dtype=torch.float32, device=device)
        tb = torch.full((e - s,), float(t), dtype=torch.float32, device=device)
        out[s:e] = net(xb, tb).double().cpu().numpy()
    net.train()
    return out


@torch.no_grad()
def field_varying_t(net, x, t, device="cpu", chunk=8192):
    """v_theta for a batch whose flow times differ per sample."""
    net.eval()
    out = np.empty_like(x)
    for s in range(0, len(x), chunk):
        e = min(s + chunk, len(x))
        xb = torch.as_tensor(x[s:e], dtype=torch.float32, device=device)
        tb = torch.as_tensor(t[s:e], dtype=torch.float32, device=device)
        out[s:e] = net(xb, tb).double().cpu().numpy()
    net.train()
    return out


def _summary(name, v):
    """Per-query summary.  Indicators carry their fraction in the median
    column, since a median of a 0/1 array is not the reported quantity."""
    v = np.asarray(v)
    if name in INDICATOR:
        return float(v.mean()), None, None
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan"), None, None
    return (float(np.median(v)), float(np.percentile(v, 25)),
            float(np.percentile(v, 75)))


def evaluate(net, P, scal, cfg, li, lam, width, seed, step, run_id,
             device="cpu", save_queries=True):
    """Rows for metrics.csv and, optionally, the per-query arrays."""
    U = P["U"]
    u_rms, vscale, s_u = scal["u_rms"], scal["vscale"], scal["s_u"]
    margin = float(cfg["escape_margin"])
    ts = [float(t) for t in cfg["t_eval"]]

    rows, arrays = [], {}
    base = dict(run_id=run_id, d=scal["d"], M=scal["M"], **{"lambda": lam},
                sigma2=scal["sigma2"][str(lam)], W=width, seed=seed, step=step)

    for ti, t in enumerate(ts):
        kap = scal["kappa_hat"][f"{lam}|{t}"]
        for name in cfg["query_sets"]:
            xk = _key(name, "X", ti) if name == "OFF" else _key(name, "X", li, ti)
            x = P[xk]
            v_star = P[_key(name, "v_star", li, ti)]
            i_star = P[_key(name, "i_star", li, ti)].astype(np.intp)
            ess = P[_key(name, "ess", li, ti)]
            rep_star = P[_key(name, "rep_star", li, ti)]

            v_th = field(net, x, t, device)
            per = {
                "r": np.linalg.norm(v_th - v_star, axis=1) / u_rms,
                "ess": ess,
                "rep_star": rep_star,
                "rep_theta": replay_indicator(v_th, U, i_star, s_u),
            }
            if name == "IN":
                per["alpha_gen"] = P[_key("IN", "alpha_gen", li, ti)]
            else:
                f = P[_key(name, "F", ti) if name == "OFF"
                      else _key(name, "F", li, ti)]
                Dt = P[_key(name, "Dt", ti) if name == "OFF"
                       else _key(name, "Dt", li, ti)]
                e_th = np.linalg.norm(v_th - f, axis=1) / vscale
                per["e_star"] = P[_key(name, "e_star", li, ti)]
                per["e_theta"] = e_th
                per["e_nn"] = P[_key(name, "e_nn", li, ti)]
                per["D_t"] = Dt
                per["escape"] = e_th < Dt - margin
                if name == "HO":
                    # The same three quantities against u*, whose floor
                    # does not move with lambda.
                    u_star = P[_key("HO", "u_star")]
                    Dt_u = P[_key("HO", "Dt_u")]
                    e_th_u = np.linalg.norm(v_th - u_star, axis=1) / vscale
                    per["e_star_u"] = P[_key("HO", "e_star_u", li, ti)]
                    per["e_theta_u"] = e_th_u
                    per["e_nn_u"] = P[_key("HO", "e_nn_u", li, ti)]
                    per["D_t_u"] = Dt_u
                    per["escape_u"] = e_th_u < Dt_u - margin

            for metric, v in per.items():
                med, q25, q75 = _summary(metric, v)
                rows.append(dict(base, kappa_hat_t=kap, t=t, set=name,
                                 metric=metric, median=med,
                                 q25=q25, q75=q75))
            if save_queries:
                for metric, v in per.items():
                    arrays[_key(name, t, metric)] = np.asarray(
                        v, dtype=np.float32)

    # ---- excess-loss diagnostic -----------------------------------------
    j = P[_key("dL", "j", li)].astype(np.intp)
    tt = P[_key("dL", "t", li)]
    per_star = P[_key("dL", "per_star", li)]
    sig = np.sqrt(scal["sigma2"][str(lam)])
    eps = _deltaL_eps(scal, li, len(j))
    C = P["C"]
    x = C[j] + tt[:, None] * U[j] + sig * eps
    v_th = field_varying_t(net, x, tt, device)
    per_theta = ((v_th - U[j]) ** 2).mean(1)
    diff = per_theta - per_star
    dL = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(diff.size))
    rows.append(dict(base, kappa_hat_t=None, t=None, set="all",
                     metric="delta_L", median=dL, q25=se, q75=None))
    rows.append(dict(base, kappa_hat_t=None, t=None, set="all",
                     metric="L_theta", median=float(per_theta.mean()),
                     q25=None, q75=None))
    return rows, arrays, dL, se


def _deltaL_eps(scal, li, n):
    from fm_replay.rng import rng as make_rng, NOISE
    from .problem import SUB_DELTAL
    r = make_rng(scal["system_id"], scal["eval_seed"], NOISE,
                 sub=(SUB_DELTAL, li))
    r.integers(0, scal["M"], n)
    r.random(n)
    return r.standard_normal((n, scal["d"]))
