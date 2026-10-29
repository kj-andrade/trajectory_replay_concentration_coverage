"""Posterior weights, ESS, and the kernel velocity estimator.

The weights are a softmax over separations, Euclidean under an isotropic
conditional path and Mahalanobis under a shared anisotropic one.  Everything
here is written so that one distance computation serves an entire bandwidth
sweep: the shape of the covariance enters the distances, the scale sigma^2
enters only as a divisor inside the softmax.
"""
from __future__ import annotations

import numpy as np


def softmax_weights(d2, sigma2):
    """Row-stabilised softmax of -d2 / (2 sigma^2)."""
    z = d2 / (-2.0 * sigma2)
    z -= z.max(1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(1, keepdims=True)


def ess(alpha):
    """Effective sample size, 1 / sum_j alpha_j^2, per query."""
    return 1.0 / np.einsum("ij,ij->i", alpha, alpha)


def training_law_logits(mu, i, eps, sigma):
    """Log-weight gaps for queries x = mu_i + sigma * eps, Section E1.

    The common term sigma^2 ||eps||^2 cancels between numerator and
    denominator, so it is never formed:

        l_j = -( ||Delta_ij||^2 + 2 sigma <Delta_ij, eps> ) / (2 sigma^2),

    with l_i = 0 by construction.  The inner products are obtained from two
    matrix products against the bank rather than by materialising Delta.
    """
    mu2 = np.einsum("ij,ij->i", mu, mu)
    # ||Delta_ij||^2 = ||mu_i||^2 + ||mu_j||^2 - 2 <mu_i, mu_j>
    d2 = mu2[i][:, None] + mu2[None, :] - 2.0 * (mu[i] @ mu.T)
    np.maximum(d2, 0.0, out=d2)
    # <Delta_ij, eps> = <mu_i, eps> - <mu_j, eps>
    me = eps @ mu.T                                  # (b, M)
    dot = me[np.arange(len(i)), i][:, None] - me
    return -(d2 + 2.0 * sigma * dot) / (2.0 * sigma ** 2), d2


def weights_from_logits(logits):
    z = logits - logits.max(1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(1, keepdims=True)


def sq_distances(X, C, W_unit=None):
    """Squared separations, Euclidean or in the whitened frame of W_unit."""
    if W_unit is not None:
        X = X @ W_unit.T
        C = C @ W_unit.T
    x2 = np.einsum("ij,ij->i", X, X)
    c2 = np.einsum("ij,ij->i", C, C)
    d2 = x2[:, None] + c2[None, :] - 2.0 * (X @ C.T)
    return np.maximum(d2, 0.0, out=d2)


def nearest(d2):
    return d2.argmin(1)


def estimate(d2, U, sigma2):
    """Kernel estimate and its ESS at one bandwidth."""
    a = softmax_weights(d2, sigma2)
    return a @ U, ess(a), a.max(1)
