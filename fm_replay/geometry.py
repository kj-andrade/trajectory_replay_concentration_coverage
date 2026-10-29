"""Pair-displacement spectrum and the shared covariance shapes of E3.

A covariance is written Sigma = sigma^2 U diag(w) U^T, splitting the SHAPE w
from the SCALE sigma^2.  Only the shape enters the distance matrix, so one
distance computation per geometry serves the whole bandwidth sweep.
"""
from __future__ import annotations

import numpy as np

EIG_FLOOR = 1e-12


def displacement_spectrum(C):
    """Gamma = 2 Cov(C), its eigenbasis, effective rank and V.

    Gamma is the second moment of the pair displacement Delta_ij between two
    independently drawn bank points, so tr Gamma = d * V.
    """
    G = np.atleast_2d(2.0 * np.cov(C, rowvar=False, ddof=1))
    lam, U = np.linalg.eigh(G)
    lam = np.clip(lam, EIG_FLOOR, None)
    d = C.shape[1]
    reff = float((np.sqrt(lam).sum() ** 2) / lam.sum())
    V = float(lam.sum() / d)
    return lam, U, reff, V


GEOMETRIES = ["iso", "halfW-trace", "halfW-kappa", "fullW-trace"]


def covariance_shape(kind, lam, U, d, reff):
    """(W_unit, kappa_coef, trace_coef) for one geometry.

    Returns the whitening map W_unit with d2_unit = ||W_unit (x - c)||^2, so
    that the Mahalanobis distance is d2_unit / sigma^2, together with

        Ekappa   = kappa_coef / sigma^2      the bank-pair mean concentration
        tr Sigma = trace_coef * sigma^2

    The four shapes separate the two things equal-trace half-whitening
    conflates: the concentration it concedes, and the anisotropy itself.

      iso          w = 1                       Ekappa = d V / sigma^2
      halfW-trace  w prop sqrt(lam), tr matched to iso
                                               Ekappa = r_eff V / sigma^2
      halfW-kappa  same shape rescaled by r_eff/d
                                               Ekappa = d V / sigma^2
      fullW-trace  w prop lam, tr matched to iso
                                               Ekappa = d V / sigma^2
    """
    s = np.sqrt(lam)
    if kind == "iso":
        w = np.ones(d)
    elif kind == "halfW-trace":
        w = d * s / s.sum()
    elif kind == "halfW-kappa":
        w = reff * s / s.sum()
    elif kind == "fullW-trace":
        w = d * lam / lam.sum()
    else:
        raise ValueError(f"unknown geometry {kind!r}")
    W_unit = (U / np.sqrt(w)).T
    return W_unit, float((lam / w).sum()), float(w.sum())


def mean_concentration_mc(C, W_unit, sigma2, rng, n_pairs=20000):
    """Monte Carlo check of Ekappa over random distinct bank pairs.

    Twenty thousand pairs rather than five: under full whitening on a
    low-rank attractor the per-pair Mahalanobis form is heavy tailed
    (sd/mean ~ 2.7 where a well-conditioned system gives ~0.15, because
    Sigma^-1 amplifies the near-null directions of an ill-conditioned
    Gamma).  Five thousand pairs give that arm a relative standard error
    near 4%, so a 2% agreement check on a single seed would fail on
    estimator variance alone rather than on any disagreement.
    """
    M = C.shape[0]
    i = rng.integers(0, M, n_pairs)
    j = rng.integers(0, M, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    Delta = C[i] - C[j]
    if W_unit is not None:
        Delta = Delta @ W_unit.T
    return float(np.einsum("ij,ij->i", Delta, Delta).mean() / sigma2)
