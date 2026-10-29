"""The fixed evaluation problem.

Everything a run is measured against is built once and cached: the bank, the
held-out windows, the query sets, the closed-form field, and the hull floor.
None of it depends on the run seed, which seeds training only, so all
thirty runs are compared on byte-identical queries.

Reuses the repository's components rather than reimplementing them:
trajectory and standardisation from ``fm_replay.data``, bank construction
from ``transition_bank``, the separation constant from the E1 module, the
displacement spectrum from ``fm_replay.geometry``, and the hull floor from
``fm_replay.coverage``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths  # noqa: F401  (puts the repository root on sys.path)
from .closed_form import (closed_form, replay_indicator, target_spacing,
                          star_loss_batch)

from fm_replay.coverage import hull_floor, check_simplex
from fm_replay.data import make_data, transition_bank, true_velocity
from fm_replay.experiments.onset import separation_constant
from fm_replay.geometry import displacement_spectrum
from fm_replay.rng import rng as make_rng, QUERY_TEST, NOISE, BANK
from fm_replay.systems import get_system

# Sub-keys for the evaluation draws.  Fixed integers, never hashes: adding a
# draw here must not shift any existing one.
SUB_HELDOUT = 910
SUB_BANKIDX = 911
SUB_OFFDIR = 912
SUB_EPS = 920
SUB_DELTAL = 930
SUB_VMC = 940

SETS = ("IN", "HO", "OFF")


def _key(*parts):
    return "|".join(str(p) for p in parts)


def _sigma_grid(d, V_half, lambdas, M):
    """sigma_lambda^2 = d V_hat / (lambda log M), natural log."""
    logM = float(np.log(M))
    return {float(L): float(d * V_half / (L * logM)) for L in lambdas}


def _heldout_windows(D, n, stride, r):
    """n windows (x*_tau, x*_{tau+1}) from the independent test trajectory,
    standardised with the training statistics."""
    from fm_replay.data import standardize
    T = D["test"]
    idx = r.choice(len(T) - stride - 1, n, replace=False)
    c = standardize(T[idx], D)
    u = (T[idx + stride] - T[idx]) / D["sd"]
    return c, u


def build(cfg, cache_dir=None, verbose=True):
    """Build (or load) the cached evaluation problem."""
    cache_dir = Path(cache_dir or paths.CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz, meta = cache_dir / "problem.npz", cache_dir / "problem.json"
    if npz.exists() and meta.exists():
        z = np.load(npz)
        return dict(z), json.loads(meta.read_text(encoding="utf-8"))

    S = get_system(cfg["system"])
    seed = int(cfg["eval_seed"])
    M, nq = int(cfg["M"]), int(cfg["n_queries"])
    d = S.d
    lambdas = [float(x) for x in cfg["lambdas"]]
    ts = [float(x) for x in cfg["t_eval"]]

    D = make_data(S, seed, paths.ROOT / "results" / "cache")

    # ---- bank and targets, k = 1 linear: mu_t = c + t u ------------------
    C, U = transition_bank(D, M, seed)
    mu0, mu1 = C, C + U
    u_rms = float(np.sqrt((U ** 2).sum(1).mean()))
    s_u = target_spacing(U)
    vscale = float(D["vscale"])

    # Off-support radius in the units the closed-form query family uses.
    _, _, reff, V_state = displacement_spectrum(C)
    off_scale = float(np.sqrt(d * V_state))
    off_radius = float(cfg["off_family"].split("-")[1]) * off_scale

    # ---- separation constant and the bandwidth grid ----------------------
    V_of_t = {t: float(separation_constant((1 - t) * mu0 + t * mu1)) for t in ts}
    t_half = float(cfg["v_hat_at"])
    V_half = float(separation_constant((1 - t_half) * mu0 + t_half * mu1))

    # Monte-Carlo cross-check of the closed form at t = v_hat_at.
    rmc = make_rng(S.id, seed, BANK, sub=(SUB_VMC,))
    npair = int(cfg["v_hat_mc_pairs"])
    i_mc, j_mc = rmc.integers(0, M, npair), rmc.integers(0, M, npair)
    keep = i_mc != j_mc
    mu_h = (1 - t_half) * mu0 + t_half * mu1
    dl = mu_h[i_mc[keep]] - mu_h[j_mc[keep]]
    V_half_mc = float(np.einsum("ij,ij->i", dl, dl).mean() / d)

    sigma2 = _sigma_grid(d, V_half, lambdas, M)
    kappa = {(L, t): d * V_of_t[t] / sigma2[L] for L in lambdas for t in ts}

    # ---- fixed query construction ----------------------------------------
    r_ho = make_rng(S.id, seed, QUERY_TEST, sub=(SUB_HELDOUT,))
    ho_c, ho_u = _heldout_windows(D, nq, S.stride, r_ho)
    ho0, ho1 = ho_c, ho_c + ho_u

    i_gen = make_rng(S.id, seed, QUERY_TEST,
                     sub=(SUB_BANKIDX,)).integers(0, M, nq)

    r_off = make_rng(S.id, seed, QUERY_TEST, sub=(SUB_OFFDIR,))
    n_hat = r_off.standard_normal((nq, d))
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)

    out = {}
    scal = dict(system=S.name, system_id=S.id, d=d, M=M, n_queries=nq,
                eval_seed=seed, stride=S.stride, dtau=S.dtau,
                u_rms=u_rms, s_u=s_u, vscale=vscale, reff=reff,
                reff_over_d=reff / d, V_state=V_state,
                off_scale=off_scale, off_radius=off_radius,
                off_family=cfg["off_family"],
                V_half=V_half, V_half_mc=V_half_mc,
                V_of_t={str(t): V_of_t[t] for t in ts},
                sigma2={str(L): sigma2[L] for L in lambdas},
                kappa_hat={f"{L}|{t}": float(kappa[(L, t)])
                           for L in lambdas for t in ts},
                lambdas=lambdas, t_eval=ts, log_M=float(np.log(M)))

    out["C"], out["U"] = C, U
    out["i_gen"] = i_gen

    # Second reference field: the held-out window's own displacement u*.
    # The primary reference is true_velocity at the query, as in the
    # closed-form coverage experiment, but u* is carried alongside because
    # it does NOT move with lambda: its floor depends only on the window and
    # the bank, so the escape question can be asked both ways (Fig. 7a).
    out[_key("HO", "u_star")] = ho_u
    Dt_u, viol_u = hull_floor(U, ho_u)
    out[_key("HO", "Dt_u")] = Dt_u / vscale

    worst_viol = check_simplex(viol_u)
    for ti, t in enumerate(ts):
        # OFF is deterministic: the closed-form off-support family carries no
        # kernel noise, so its queries depend on t but not on lambda.
        x_off = (1 - t) * ho0 + t * ho1 + off_radius * n_hat
        f_off = true_velocity(x_off, D)
        Dt_off, viol = hull_floor(U, f_off)
        worst_viol = max(worst_viol, check_simplex(viol))
        out[_key("OFF", "X", ti)] = x_off
        out[_key("OFF", "F", ti)] = f_off
        out[_key("OFF", "Dt", ti)] = Dt_off / vscale

        for li, L in enumerate(lambdas):
            s2 = sigma2[L]
            sig = np.sqrt(s2)

            eps_in = make_rng(S.id, seed, NOISE,
                              sub=(SUB_EPS, li, 0, ti)).standard_normal((nq, d))
            eps_ho = make_rng(S.id, seed, NOISE,
                              sub=(SUB_EPS, li, 1, ti)).standard_normal((nq, d))

            x_in = (1 - t) * mu0[i_gen] + t * mu1[i_gen] + sig * eps_in
            x_ho = (1 - t) * ho0 + t * ho1 + sig * eps_ho
            f_ho = true_velocity(x_ho, D)
            Dt_ho, viol = hull_floor(U, f_ho)
            worst_viol = max(worst_viol, check_simplex(viol))

            out[_key("IN", "X", li, ti)] = x_in
            out[_key("HO", "X", li, ti)] = x_ho
            out[_key("HO", "F", li, ti)] = f_ho
            out[_key("HO", "Dt", li, ti)] = Dt_ho / vscale

            for name, x in (("IN", x_in), ("HO", x_ho), ("OFF", x_off)):
                v_star, alpha, ess = closed_form(x, t, mu0, mu1, U, s2)
                i_star = alpha.argmax(1)
                out[_key(name, "v_star", li, ti)] = v_star
                out[_key(name, "i_star", li, ti)] = i_star
                out[_key(name, "ess", li, ti)] = ess
                out[_key(name, "rep_star", li, ti)] = replay_indicator(
                    v_star, U, i_star, s_u)
                if name == "IN":
                    out[_key("IN", "alpha_gen", li, ti)] = alpha[
                        np.arange(nq), i_gen]
                else:
                    f = f_ho if name == "HO" else f_off
                    out[_key(name, "e_star", li, ti)] = (
                        np.linalg.norm(v_star - f, axis=1) / vscale)
                    out[_key(name, "e_nn", li, ti)] = (
                        np.linalg.norm(U[i_star] - f, axis=1) / vscale)
                    if name == "HO":
                        out[_key("HO", "e_star_u", li, ti)] = (
                            np.linalg.norm(v_star - ho_u, axis=1) / vscale)
                        out[_key("HO", "e_nn_u", li, ti)] = (
                            np.linalg.norm(U[i_star] - ho_u, axis=1) / vscale)
        if verbose:
            print(f"  [problem] t={t} done", flush=True)

    # ---- excess-loss reference --------------------------------------------
    nB = int(cfg["excess_loss_samples"])
    for li, L in enumerate(lambdas):
        r = make_rng(S.id, seed, NOISE, sub=(SUB_DELTAL, li))
        j = r.integers(0, M, nB)
        tt = r.random(nB)
        eps = r.standard_normal((nB, d))
        Lstar, per = star_loss_batch(mu0, mu1, U, sigma2[L], j, tt, eps)
        out[_key("dL", "j", li)] = j.astype(np.int32)
        out[_key("dL", "t", li)] = tt
        out[_key("dL", "per_star", li)] = per
        scal.setdefault("L_star", {})[str(L)] = Lstar
        if verbose:
            print(f"  [problem] excess-loss reference lambda={L:g} "
                  f"L(v*)={Lstar:.6f}", flush=True)

    scal["max_simplex_violation"] = float(worst_viol)
    np.savez_compressed(npz, **out)
    meta.write_text(json.dumps(scal, indent=2), encoding="utf-8")
    return out, scal


def eval_noise(scal, li, si, ti):
    """Regenerate the fixed evaluation noise for (lambda, set, t)."""
    return make_rng(scal["system_id"], scal["eval_seed"], NOISE,
                    sub=(SUB_EPS, li, si, ti)).standard_normal(
        (scal["n_queries"], scal["d"]))


def deltaL_batch(P, scal, li):
    """Regenerate the fixed excess-loss batch for one lambda."""
    d, M = scal["d"], scal["M"]
    r = make_rng(scal["system_id"], scal["eval_seed"], NOISE,
                 sub=(SUB_DELTAL, li))
    j = r.integers(0, M, len(P[_key("dL", "j", li)]))
    tt = r.random(len(j))
    eps = r.standard_normal((len(j), d))
    return j, tt, eps
