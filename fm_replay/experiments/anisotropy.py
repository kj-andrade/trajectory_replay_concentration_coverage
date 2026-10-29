"""E3: covariance geometry at matched concentration.

Two questions, deliberately separated:

  1. At equal bank-pair mean concentration kappa_bar, does the SHAPE of a
     shared covariance change the error?  The comparison is paired: the same
     bank and the same queries, differing only in geometry.

  2. At equal trace, the concentration-optimal covariance Sigma* prop
     Gamma^{1/2} of Proposition 5 reduces kappa_bar by d / r_eff.  What does
     that concession cost, and how much of it does the shape refund?

The second question is answered by three numbers at each bandwidth:

  e1 = iso at kappa_0                    the equal-trace isotropic arm
  e2 = halfW-trace at kappa_0            Proposition 5's covariance, whose
                                         realised concentration is
                                         (r_eff/d) kappa_0
  e3 = iso at (r_eff/d) kappa_0          the isotropic arm at the SAME
                                         realised concentration as e2

so that psi = (e3 - e2)/(e3 - e1) is the fraction of the concession the
anisotropic shape gives back.
"""
from __future__ import annotations

import numpy as np

from ..data import make_data, transition_bank, true_velocity
from ..geometry import (displacement_spectrum, covariance_shape,
                        mean_concentration_mc, GEOMETRIES)
from ..kernel import sq_distances, estimate
from ..metrics import per_seed
from ..queries import build_families
from ..rng import rng as make_rng
from ..systems import get_system

MATCHED_GEOMS = ["halfW-kappa", "fullW-trace"]   # same kappa_bar as iso
RATIO_GEOM = "halfW-kappa"                       # the ratio plotted in Fig 3a


def _kappa_grid(spec, M):
    hi = float(spec["hi_mult"]) * 2.0 * np.log(M)
    return list(np.geomspace(float(spec["lo"]), hi, int(spec["n"])))


def _err(d2, U, sigma2, v_true, vscale):
    v_hat, _, _ = estimate(d2, U, sigma2)
    return np.linalg.norm(v_hat - v_true, axis=1) / vscale


def run(cfg, cache=None):
    rows = []
    nq = cfg["n_queries"]
    families = cfg["families"]
    M = cfg["M"]
    kappas = _kappa_grid(cfg["kappa_grid"], M)

    for name in cfg["systems"]:
        S = get_system(name)
        for seed in cfg["seeds"]:
            D = make_data(S, seed, cache)
            C, U = transition_bank(D, M)
            lam, Uvec, reff, V = displacement_spectrum(C)
            scale = np.sqrt(S.d * V)
            ratio = reff / S.d

            shapes, checks = {}, {}
            for g in GEOMETRIES:
                W_unit, kcoef, tcoef = covariance_shape(g, lam, Uvec, S.d, reff)
                shapes[g] = (W_unit if g != "iso" else None, kcoef, tcoef)
                # Verify the closed-form kappa_bar against random bank pairs.
                # A fresh generator per geometry, seeded identically, so every
                # geometry is checked on the SAME pairs.
                checks[g] = mean_concentration_mc(
                    C, shapes[g][0], 1.0,
                    make_rng(S.id, seed, "bank", sub=(77,))) / kcoef

            rows.append(dict(block="system", system=name, system_id=S.id,
                             d=S.d, M=M, seed=seed, V=V, reff=reff,
                             reff_over_d=ratio,
                             **{f"mc_check_{g}": checks[g] for g in GEOMETRIES}))

            qs = build_families(D, C, scale, nq, "test", families)
            for fam in families:
                X = qs[fam]
                v_true = true_velocity(X, D)
                d2 = {g: sq_distances(X, C, shapes[g][0]) for g in GEOMETRIES}

                for kappa0 in kappas:
                    err = {}
                    # One scale serves every geometry: sigma^2 = d V / kappa_0.
                    # The shapes then differ in what that scale buys.  iso,
                    # halfW-kappa and fullW-trace all realise kappa_bar =
                    # kappa_0; halfW-trace spends the same trace as iso and
                    # so realises (r_eff/d) kappa_0 instead.
                    sigma2 = S.d * V / kappa0
                    for g in GEOMETRIES:
                        _, kcoef, tcoef = shapes[g]
                        e = _err(d2[g], U, sigma2, v_true, D["vscale"])
                        err[g] = per_seed(e)
                        rows.append(dict(
                            block="arm", system=name, system_id=S.id, d=S.d,
                            M=M, seed=seed, family=fam, geometry=g,
                            kappa_target=float(kappa0),
                            Ekappa=float(kcoef / sigma2),
                            trS_over_d=float(tcoef * sigma2 / S.d),
                            sigma2=float(sigma2), nerr=err[g]))

                    # (1) paired shape ratio at matched kappa_bar
                    for g in MATCHED_GEOMS:
                        rows.append(dict(
                            block="ratio", system=name, system_id=S.id,
                            d=S.d, M=M, seed=seed, family=fam, geometry=g,
                            kappa_target=float(kappa0),
                            R=err[g] / err["iso"] if err["iso"] > 0 else np.nan))

                    # (2) equal-trace decomposition
                    k_realised = ratio * kappa0
                    e3 = _err(d2["iso"], U, S.d * V / k_realised, v_true,
                              D["vscale"])
                    e1, e2 = err["iso"], err["halfW-trace"]
                    e3m = per_seed(e3)
                    denom = e3m - e1
                    rows.append(dict(
                        block="equal_trace", system=name, system_id=S.id,
                        d=S.d, M=M, seed=seed, family=fam,
                        kappa_target=float(kappa0),
                        kappa_realised=float(k_realised),
                        e1=e1, e2=e2, e3=e3m,
                        psi=(e3m - e2) / denom if abs(denom) > 1e-12 else np.nan,
                        onset=float(2 * np.log(M))))
        print(f"  [E3] {name} done", flush=True)
    return rows
