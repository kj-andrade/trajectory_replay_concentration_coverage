"""The closed-form empirical minimiser v*, written out directly in float64.

Kept separate from ``fm_replay.kernel`` on purpose: the tests check one
implementation against the other rather than against itself.
"""
from __future__ import annotations

import numpy as np


def closed_form(x, t, mu0, mu1, U, sigma2):
    """v*, alpha and ESS at flow time t for queries x.

    x: (Q, d); mu0, mu1: (M, d) window endpoints; U: (M, d) targets.
    """
    mu_t = (1 - t) * mu0 + t * mu1                       # (M, d)
    sq = ((x ** 2).sum(1, keepdims=True) - 2 * x @ mu_t.T
          + (mu_t ** 2).sum(1)[None, :])
    logits = -sq / (2 * sigma2)
    logits -= logits.max(1, keepdims=True)               # logsumexp-stable
    alpha = np.exp(logits)
    alpha /= alpha.sum(1, keepdims=True)
    v_star = alpha @ U
    ess = 1.0 / (alpha ** 2).sum(1)
    return v_star, alpha, ess


def nearest_target(v, U):
    """argmin_j ||v - u_j||, in chunks so the (Q, M) block stays small."""
    u2 = np.einsum("ij,ij->i", U, U)
    out = np.empty(len(v), dtype=np.intp)
    for s in range(0, len(v), 4096):
        e = min(s + 4096, len(v))
        d2 = (np.einsum("ij,ij->i", v[s:e], v[s:e])[:, None]
              - 2.0 * (v[s:e] @ U.T) + u2[None, :])
        out[s:e] = d2.argmin(1)
    return out


def replay_indicator(v, U, i_star, s_u):
    """rep(v): the estimate is the stored target of the dominant component,
    and is closer to it than half the typical spacing between targets."""
    nn = nearest_target(v, U)
    close = np.linalg.norm(v - U[i_star], axis=1) < 0.5 * s_u
    return (nn == i_star) & close


def target_spacing(U):
    """s_u = median_j min_{l != j} ||u_j - u_l||."""
    M = U.shape[0]
    u2 = np.einsum("ij,ij->i", U, U)
    best = np.empty(M)
    for s in range(0, M, 1024):
        e = min(s + 1024, M)
        d2 = u2[s:e, None] - 2.0 * (U[s:e] @ U.T) + u2[None, :]
        np.maximum(d2, 0.0, out=d2)
        d2[np.arange(e - s), np.arange(s, e)] = np.inf
        best[s:e] = np.sqrt(d2.min(1))
    return float(np.median(best))


def conditional_loss(v, U, j):
    """L(v) = mean over batch and dims of (v - u_j)^2, and the per-sample
    contribution, which is what the paired standard error needs."""
    per = ((v - U[j]) ** 2).mean(1)
    return float(per.mean()), per


def star_loss_batch(mu0, mu1, U, sigma2, j, t, eps, chunk=2048):
    """Per-sample loss of v* on a training batch drawn as in training.

    For k = 1 and the linear interpolant mu_t = c + t u, so the (B, M)
    separation matrix is assembled from two matrix products and three
    per-component scalars rather than by materialising mu_t for every
    sample's own t.
    """
    d = U.shape[1]
    c2 = np.einsum("ij,ij->i", mu0, mu0)
    du = mu1 - mu0
    cu = np.einsum("ij,ij->i", mu0, du)
    u2 = np.einsum("ij,ij->i", du, du)

    per = np.empty(len(j))
    for s in range(0, len(j), chunk):
        e = min(s + chunk, len(j))
        ts = t[s:e][:, None]
        x = mu0[j[s:e]] + t[s:e][:, None] * du[j[s:e]] + np.sqrt(sigma2) * eps[s:e]
        sq = (np.einsum("ij,ij->i", x, x)[:, None]
              - 2.0 * (x @ mu0.T) - 2.0 * ts * (x @ du.T)
              + c2[None, :] + 2.0 * ts * cu[None, :] + (ts ** 2) * u2[None, :])
        logits = -sq / (2 * sigma2)
        logits -= logits.max(1, keepdims=True)
        a = np.exp(logits)
        a /= a.sum(1, keepdims=True)
        per[s:e] = ((a @ U - U[j[s:e]]) ** 2).mean(1)
    return float(per.mean()), per
