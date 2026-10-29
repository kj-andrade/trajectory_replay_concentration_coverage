"""Trajectories, standardisation, transition banks and exact ground truth.

Everything downstream works in standardised
coordinates: the training mean and standard deviation are the only quantities
carried back to raw space, and they are carried only to evaluate the true
flow map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .integrate import integrate, flow_map
from .rng import rng as make_rng, TRAJ, TEST_TRAJ, BANK

BURN = 30_000
GROUND_TRUTH_SUBSTEPS = 40
VSCALE_SAMPLES = 2000

# Default training-trajectory lengths.
L_TRAIN_DEFAULT = {0: 400_000}          # L63
L_TRAIN_L96 = 200_000                   # fixed L96 systems, ids 1-5
L_TRAIN_ONSET = 100_000                 # onset sweep, ids 10+


def default_L_train(system):
    if system.id in L_TRAIN_DEFAULT:
        return L_TRAIN_DEFAULT[system.id]
    return L_TRAIN_ONSET if system.id >= 10 else L_TRAIN_L96


def standardize(x, D):
    return (x - D["mean"]) / D["sd"]


def unstandardize(x, D):
    return x * D["sd"] + D["mean"]


def _spin_up(system, x0, seed):
    """Burn in, with the per-seed perturbation the deterministic ICs need."""
    from .systems import DETERMINISTIC_IC
    if system.id in DETERMINISTIC_IC:
        # The initial condition selects the attractor and must not be
        # randomised; per-seed variation enters here instead.
        x0 = x0 + 1e-3 * make_rng(system.id, seed, TRAJ).standard_normal(system.d)
    return integrate(x0, BURN, system.dt, system.rhs, system.params)[-1]


def _simulate(system, L, seed, cache_dir):
    """Training and test trajectories, cached on disk."""
    cache = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"{system.name}_L{L}_seed{seed}.npz"
        if cache.exists():
            z = np.load(cache)
            return z["train"], z["test"]

    x0 = system.ic(system.d, system.params, make_rng(system.id, seed, TRAJ))
    train = integrate(_spin_up(system, x0, seed), L, system.dt,
                      system.rhs, system.params)

    y0 = system.ic(system.d, system.params,
                   make_rng(system.id, seed, TEST_TRAJ))
    y0 = integrate(y0, BURN, system.dt, system.rhs, system.params)[-1]
    test = integrate(y0, max(L // 4, 20_000), system.dt,
                     system.rhs, system.params)

    if cache is not None:
        np.savez_compressed(cache, train=train, test=test)
    return train, test


_MEM = {}
_MEM_MAX = 3


def make_data(system, seed, cache_dir=None, L_train=None):
    """Standardised trajectories plus the error normalisation v_scale.

    Memoised on (system, seed, L): an experiment revisits the same record
    once per window order, interpolant and bandwidth, and re-reading a
    100 MB archive and re-deriving v_scale each time dominates the run.
    The cache is deliberately shallow, since each record is large.
    """
    key = (system.name, int(seed),
           L_train if L_train is not None else default_L_train(system))
    if key in _MEM:
        return _MEM[key]
    D = _make_data(system, seed, cache_dir, L_train)
    if len(_MEM) >= _MEM_MAX:
        _MEM.pop(next(iter(_MEM)))
    _MEM[key] = D
    return D


def _make_data(system, seed, cache_dir=None, L_train=None):
    if system.kind == "gauss":
        return dict(system=system, seed=seed, d=system.d, kind="gauss",
                    mean=np.zeros(system.d), sd=np.ones(system.d))

    L = default_L_train(system) if L_train is None else L_train
    train, test = _simulate(system, L, seed, cache_dir)

    sd = train.std(0)
    sd[sd < 1e-8] = 1.0
    D = dict(system=system, seed=seed, d=system.d, kind="ode",
             train=train, test=test, mean=train.mean(0), sd=sd,
             L_train=L, dtau=system.dtau)

    # v_scale normalises every error in the study: the median speed of the
    # true field on the test trajectory.
    idx = make_rng(system.id, seed, TEST_TRAJ).choice(
        len(test) - 1, min(VSCALE_SAMPLES, len(test) - 1), replace=False)
    X = standardize(test[idx], D)
    D["vscale"] = float(np.median(np.linalg.norm(true_velocity(X, D), axis=1)))
    return D


def true_velocity(X, D, substeps=GROUND_TRUTH_SUBSTEPS):
    """f(x): the exact one-step displacement of the true flow, standardised.

    Defined everywhere, including off the attractor, which is what makes the
    off-manifold query families well posed.
    """
    S = D["system"]
    raw = unstandardize(X, D)
    nxt = flow_map(raw, D["dtau"], substeps, S.rhs, S.params)
    return (nxt - raw) / D["sd"]


def bank_indices(D, M, rng):
    """Nested bank draws.

    A single permutation per (system, seed) is truncated to M, so banks of
    different sizes are nested and comparisons across M are paired.
    """
    n = len(D["train"]) - D["system"].stride - 1
    return rng.permutation(n)[:M]


def transition_bank(D, M, seed=None):
    """Pairwise transition bank (c_j, u_j) in standardised coordinates."""
    S = D["system"]
    r = make_rng(S.id, D["seed"] if seed is None else seed, BANK)
    idx = bank_indices(D, M, r)
    c = standardize(D["train"][idx], D)
    u = (D["train"][idx + S.stride] - D["train"][idx]) / D["sd"]
    return c, u


def gauss_bank(system, seed, M):
    """iid N(0, I_d) control cloud; no targets, no ground truth."""
    return make_rng(system.id, seed, BANK).standard_normal((M, system.d))
