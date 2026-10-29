"""E2: target coverage and the attainable error floor.

On the pairwise-transition bank, three quantities are measured per query and
set against a coverage proxy:

  D_t(x)   the convex-hull representation floor of Theorem 4, which no
           bandwidth and no shared covariance can cross;
  e_best   the error of the best isotropic Gaussian-kernel operating point,
           selected on a validation query set and reported on a disjoint
           test set;
  e_NN     the nearest-neighbour error, the kappa -> infinity limit.

The bandwidth grid is parameterised by the bank-pair mean concentration
kappa_bar rather than by sigma, because a fixed sigma is a different regime
in every system.
"""
from __future__ import annotations

import numpy as np

from ..config import geomspace
from ..coverage import (hull_floor, check_simplex, coverage_proxy,
                        lipschitz_estimate)
from ..data import make_data, transition_bank, true_velocity
from ..geometry import displacement_spectrum
from ..kernel import sq_distances, estimate, nearest
from ..metrics import per_seed
from ..queries import build_families
from ..systems import get_system


def _kappa_grid(spec, M):
    """Log-spaced kappa_bar, with the upper end pinned to a multiple of the
    onset scale 2 log M so that every bank size is swept equally deep."""
    hi = float(spec["hi_mult"]) * 2.0 * np.log(M)
    return list(np.geomspace(float(spec["lo"]), hi, int(spec["n"])))


def _errors_over_kappa(d2, U, V, d, kappas, v_true, vscale):
    """Normalised error at each bandwidth, plus the 1-NN limit."""
    out = {}
    for kappa in kappas:
        sigma2 = d * V / kappa
        v_hat, _, _ = estimate(d2, U, sigma2)
        out[float(kappa)] = np.linalg.norm(v_hat - v_true, axis=1) / vscale
    v_nn = U[nearest(d2)]
    out[np.inf] = np.linalg.norm(v_nn - v_true, axis=1) / vscale
    return out


def run(cfg, cache=None):
    rows = []
    per_query = {}
    nq = cfg["n_queries"]
    families = cfg["families"]

    for name, M_list in cfg["grid"]:
        S = get_system(name)
        for M in M_list:
            kappas = _kappa_grid(cfg["kappa_grid"], M)
            for seed in cfg["seeds"]:
                D = make_data(S, seed, cache)
                C, U = transition_bank(D, M)
                lam, _, reff, V = displacement_spectrum(C)
                scale = np.sqrt(S.d * V)
                L_hat = lipschitz_estimate(C, U)

                qs = {split: build_families(D, C, scale, nq, split, families)
                      for split in ("val", "test")}

                for fam in families:
                    Xv, Xt = qs["val"][fam], qs["test"][fam]
                    fv, ft = true_velocity(Xv, D), true_velocity(Xt, D)
                    d2v = sq_distances(Xv, C)
                    d2t = sq_distances(Xt, C)

                    ev = _errors_over_kappa(d2v, U, V, S.d, kappas, fv,
                                            D["vscale"])
                    et = _errors_over_kappa(d2t, U, V, S.d, kappas, ft,
                                            D["vscale"])

                    # Operating point chosen on validation, reported on the
                    # disjoint test set.  The 1-NN limit is a candidate, so
                    # the sweep is not silently truncated below the optimum.
                    finite = [k for k in ev if np.isfinite(k)]
                    kstar = min(ev, key=lambda k: per_seed(ev[k]))
                    # A cell sitting at the edge of the grid has not shown
                    # its optimum, and is flagged rather than reported as if
                    # it had.
                    at_boundary = bool(not np.isfinite(kstar)
                                       or np.isclose(kstar, max(finite)))
                    e_best = et[kstar]

                    d_nn = np.sqrt(d2t.min(1))
                    rho = coverage_proxy(d_nn, S.d, V)

                    Dt, viol = hull_floor(U, ft)
                    worst = check_simplex(viol)
                    Dt_n = Dt / D["vscale"]

                    med_best = per_seed(e_best)
                    med_Dt = per_seed(Dt_n)
                    rows.append(dict(
                        block="cell", system=name, system_id=S.id, d=S.d,
                        M=M, seed=seed, family=fam, V=V, reff=reff,
                        reff_over_d=reff / S.d, vscale=D["vscale"],
                        rho=per_seed(rho), d_nn=per_seed(d_nn),
                        D_t=med_Dt, e_best=med_best,
                        e_nn=per_seed(et[np.inf]),
                        lipschitz_bound=per_seed(L_hat * d_nn / D["vscale"]),
                        phi=med_Dt / med_best if med_best > 0 else np.nan,
                        kappa_star=float(kstar),
                        kappa_star_at_boundary=at_boundary,
                        onset=float(2 * np.log(M)),
                        simplex_violation=worst))

                    # The whole error-versus-kappa curve, for the record.
                    for k, e in et.items():
                        rows.append(dict(
                            block="curve", system=name, system_id=S.id,
                            d=S.d, M=M, seed=seed, family=fam,
                            kappa=float(k), nerr=per_seed(e)))

                    if cfg["save_per_query"]:
                        tag = f"{name}|M{M}|s{seed}|{fam}"
                        per_query[f"{tag}|D_t"] = Dt_n.astype(np.float32)
                        per_query[f"{tag}|e_best"] = e_best.astype(np.float32)
                        per_query[f"{tag}|e_nn"] = et[np.inf].astype(np.float32)
                        per_query[f"{tag}|rho"] = rho.astype(np.float32)
            print(f"  [E2] {name} M={M} done", flush=True)
    return rows, per_query
