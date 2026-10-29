"""Windows and interpolants (E1).

A window is k+1 consecutive stored states, W_tau = (c_tau, ..., c_{tau+k*s}),
with knots at t_m = m/k.  An interpolant places one point mu_t(W) on each
window at flow time t; the collapse statements are properties of the
resulting cloud {mu_j}.

Two interpolants, because the theory claims the collapse exponent is
interpolant-invariant while the separation constant V_k is not:

  linear  piecewise-linear through the knots
  cubic   natural cubic spline through the knots (k >= 2 only)
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .data import standardize
from .rng import rng as make_rng, BANK


def window_bank(D, k, M, seed=None, stride=None):
    """M windows of k+1 stored states, shape (M, k+1, d).

    Start indices come from the same nested permutation the transition bank
    uses, so a window bank and a transition bank of the same size are drawn
    from one shared ordering.
    """
    S = D["system"]
    st = S.stride if stride is None else stride
    T = D["train"]
    n = len(T) - k * st - 1
    if n < 1:
        raise ValueError(f"trajectory too short for k={k}, stride={st}")
    r = make_rng(S.id, D["seed"] if seed is None else seed, BANK)
    idx = r.permutation(n)[:min(M, n)]
    offsets = np.arange(k + 1) * st
    return standardize(T[idx[:, None] + offsets[None, :]], D)


def linear_positions(W, t):
    """Piecewise-linear interpolant and its derivative at flow time t."""
    k = W.shape[1] - 1
    i = min(max(int(np.floor(t * k)), 0), k - 1)
    s = t * k - i
    mu = (1.0 - s) * W[:, i, :] + s * W[:, i + 1, :]
    # d/dt over the active segment: the knot spacing in flow time is 1/k.
    dmu = k * (W[:, i + 1, :] - W[:, i, :])
    return mu, dmu


def cubic_positions(W, t):
    """Natural cubic spline through the knots, and its derivative at t."""
    k = W.shape[1] - 1
    if k < 2:
        raise ValueError("the cubic interpolant needs k >= 2")
    knots = np.arange(k + 1) / k
    # CubicSpline vectorises over the trailing axes, so one fit serves the
    # whole bank: axis 0 is the knot axis, (M, d) ride along.
    cs = CubicSpline(knots, np.transpose(W, (1, 0, 2)), axis=0,
                     bc_type="natural")
    return np.asarray(cs(t)), np.asarray(cs(t, 1))


INTERPOLANTS = {"linear": linear_positions, "cubic": cubic_positions}


def interpolate(W, t, kind):
    """(mu, mu_dot) at flow time t for the named interpolant."""
    return INTERPOLANTS[kind](W, t)


def valid_interpolants(k):
    return ["linear"] if k < 2 else ["linear", "cubic"]
