"""E1: replay onset and dimensional scaling.

Under the training-law query model of Theorem 1, three questions:

  1. are the posterior-weight statistics functions of the single predicted
     quantity kappa_hat = d V_k / sigma^2, across ambient dimension, window
     order and interpolant?
  2. do pairwise separations concentrate at rate d^{-1/2}?
  3. how does the ESS = 2 onset scale with bank size M?

Queries are x = mu_i + sigma eps with (i, eps) held fixed across the
bandwidth sweep, so every point on a kappa curve is a common-random-number
comparison rather than an independent draw.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..config import geomspace
from ..data import make_data, gauss_bank
from ..kernel import training_law_logits, weights_from_logits, ess
from ..metrics import crossing, linear_fit, per_seed
from ..rng import rng as make_rng, QUERY_TEST, NOISE
from ..systems import gauss_system, onset_system
from ..windows import window_bank, interpolate, valid_interpolants

N_PAIRS = 5000          # bank pairs for the separation-spread statistic
KS_AT = 3               # kappa values at which the gap law is unit-tested


def separation_constant(mu):
    """V_k = (2/d) tr Cov(mu), with ddof=1.

    Equal to the mean of ||mu_i - mu_j||^2 / d over distinct bank pairs.
    """
    d = mu.shape[1]
    return float(2.0 * np.trace(np.atleast_2d(np.cov(mu, rowvar=False, ddof=1))) / d)


def separation_spread(mu, r, n_pairs=N_PAIRS):
    """sd/mean of the pairwise squared separation.

    The ratio is free of sigma, so it measures concentration of the
    separations themselves; the prediction is that it falls like d^{-1/2}.
    """
    M = mu.shape[0]
    i = r.integers(0, M, n_pairs)
    j = r.integers(0, M, n_pairs)
    keep = i != j
    Delta = mu[i[keep]] - mu[j[keep]]
    s2 = np.einsum("ij,ij->i", Delta, Delta)
    return float(s2.std(ddof=1) / s2.mean()), float(s2.mean())


def _cloud(spec, seed, cfg, cache, t):
    """The interpolant cloud mu for one (system, k, interpolant) cell."""
    kind, d, k, interp, M = spec
    if kind == "gauss":
        S = gauss_system(d)
        return S, gauss_bank(S, seed, M)
    S = onset_system(d)
    D = make_data(S, seed, cache, L_train=cfg.get("L_train"))
    W = window_bank(D, k, M, seed)
    mu, _ = interpolate(W, t, interp)
    return S, mu


def _sweep(mu, kappas, nq, S, seed, ks_idx=()):
    """ESS and 1 - alpha_i along the bandwidth sweep, plus the gap-law test."""
    M, d = mu.shape
    Vk = separation_constant(mu)

    # Common random numbers: one (i, eps) pair reused at every bandwidth.
    i = make_rng(S.id, seed, QUERY_TEST).integers(0, M, size=nq)
    eps = make_rng(S.id, seed, NOISE).standard_normal((nq, d))

    out = []
    ks = {}
    for n, kappa in enumerate(kappas):
        sigma2 = d * Vk / kappa
        logits, d2 = training_law_logits(mu, i, eps, np.sqrt(sigma2))
        a = weights_from_logits(logits)
        alpha_i = a[np.arange(nq), i]
        out.append(dict(kappa=float(kappa), sigma2=float(sigma2),
                        ess=per_seed(ess(a)),
                        one_minus_alpha=per_seed(1.0 - alpha_i)))
        if n in ks_idx:
            ks[float(kappa)] = _gap_law_ks(
                logits, d2, sigma2, i,
                make_rng(S.id, seed, NOISE, sub=(1, n)))
    return Vk, out, ks


def _gap_law_ks(logits, d2, sigma2, i, r):
    """Implementation check of Theorem 1.

    G_j = l_i - l_j should be N(kappa_ij/2, kappa_ij), so

        Z_j = (G_j - kappa_ij/2) / sqrt(kappa_ij) = <Delta_ij, eps>/||Delta_ij||

    must be standard normal.  A failure here is a bug in the estimator, not
    evidence about the theory.

    Exactly ONE competitor is taken per query.  All M-1 competitors of a
    given query are projections of the SAME eps onto different directions,
    so pooling them would feed strongly dependent values to a test that
    assumes independence and return a vanishing p-value whatever the
    estimator does.  One competitor per query keeps the sample iid across
    queries, which is what the test needs.
    """
    nq = logits.shape[0]
    j = r.integers(0, logits.shape[1], size=nq)
    clash = j == i
    j[clash] = (j[clash] + 1) % logits.shape[1]

    q = np.arange(nq)
    G = logits[q, i] - logits[q, j]
    kappa = d2[q, j] / sigma2
    ok = kappa > 1e-9
    Z = (G[ok] - kappa[ok] / 2.0) / np.sqrt(kappa[ok])
    if Z.size < 8:
        return float("nan"), float("nan"), float("nan")
    return (float(stats.kstest(Z, "norm").pvalue),
            float(Z.mean()), float(Z.std(ddof=1)))


def run(cfg, cache=None):
    """Rows for the three panels of Fig. 2 in the paper."""
    rows = []
    kappas = geomspace(cfg["kappa_grid"])
    ks_idx = np.linspace(0, len(kappas) - 1, KS_AT).astype(int).tolist()
    nq = cfg["n_queries"]

    # -- panels (a) and (b): dimension, window order, interpolant ------
    M_main = cfg["M_main"]
    flow_times = cfg.get("flow_times") or [cfg["flow_time"]]
    for t in flow_times:
        for kind, dims in (("l96", cfg["dims_l96"]), ("gauss", cfg["dims_gauss"])):
            for d in dims:
                orders = [None] if kind == "gauss" else cfg["k_orders"]
                # Seed is the OUTER loop so that one trajectory record serves
                # every window order and interpolant before the next seed is
                # touched.  Cycling seeds innermost thrashes the memo cache and
                # re-reads a hundred-megabyte archive per cell.  Every draw is
                # keyed by (system, seed), never by iteration order, so this
                # reordering cannot change a single number.
                for seed in cfg["seeds"]:
                    for k in orders:
                        interps = (["none"] if kind == "gauss"
                                   else valid_interpolants(k))
                        for interp in interps:
                            S, mu = _cloud((kind, d, k, interp, M_main), seed,
                                           cfg, cache, t)
                            Vk, sweep, ks = _sweep(mu, kappas, nq, S, seed, ks_idx)
                            spread, mean_s2 = separation_spread(
                                mu, make_rng(S.id, seed, "bank", sub=(99,)))
                            base = dict(block="sweep", system=S.name, t=t,
                                        system_id=S.id, kind=kind, d=d,
                                        k=(k if k is not None else ""),
                                        interpolant=interp, M=M_main, seed=seed,
                                        Vk=Vk, spread=spread,
                                        mean_sep2_over_d=mean_s2 / d)
                            for rec in sweep:
                                rows.append(dict(base, **rec))
                            for kap, (p, zm, zs) in ks.items():
                                rows.append(dict(base, block="ks", kappa=kap,
                                                 ks_pvalue=p, z_mean=zm,
                                                 z_sd=zs))
                print(f"  [E1] t={t} {kind} d={d} done", flush=True)

    # -- panel (c): the ESS = 2 onset against bank size ----------------
    for name in cfg["onset_systems"]:
        for seed in cfg["seeds"]:
            stars = []
            for M in cfg["M_onset"]:
                kind = "gauss" if name.startswith("Gauss") else "l96"
                d = int(name.split("-")[1])
                k = cfg["k_orders"][0] if kind == "l96" else None
                interp = "linear" if kind == "l96" else "none"
                S, mu = _cloud((kind, d, k, interp, M), seed, cfg, cache,
                               cfg["flow_time"])
                Vk, sweep, _ = _sweep(mu, kappas, nq, S, seed)
                kstar = crossing([r["kappa"] for r in sweep],
                                 [r["ess"] for r in sweep], cfg["ess_level"])
                stars.append((M, kstar))
                rows.append(dict(block="onset", system=S.name, system_id=S.id,
                                 kind=kind, d=d, k=(k if k is not None else ""),
                                 interpolant=interp, M=M, seed=seed, Vk=Vk,
                                 kappa_star=kstar,
                                 ess_level=cfg["ess_level"],
                                 two_log_M=float(2 * np.log(M))))
            c, b, r2 = linear_fit([np.log(M) for M, _ in stars],
                                  [s for _, s in stars])
            rows.append(dict(block="onset_fit", system=name, seed=seed,
                             slope_c=c, intercept_b=b, r2=r2))
        print(f"  [E1] onset {name} done", flush=True)
    return rows
