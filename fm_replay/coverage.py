"""Target coverage: the convex-hull representation floor (E2).

Theorem 4 says the infinite-capacity empirical velocity lies in the convex
hull of the stored targets, so for every query

    D_t(x) = min_{alpha in simplex} || f(x) - sum_j alpha_j u_j ||

is a lower bound on the error that no bandwidth and no shared covariance can
cross.  D_t is a property of the bank and the reference field alone.

The minimisation is a projection onto a convex hull, solved as a
non-negative least squares problem with the simplex constraint imposed by a
heavily weighted extra row:

    A = [ U^T ; lambda 1^T ],   b = [ f(x) ; lambda ].

The constraint is soft, so the realised |sum alpha - 1| is recorded and
asserted against a tolerance rather than assumed.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

LAMBDA = 1e4
MAXITER_MULT = 20
SIMPLEX_TOL = 1e-3


def hull_floor(U, F, lam=LAMBDA, maxiter_mult=MAXITER_MULT):
    """D_t for each row of F against the target cloud U (M, d).

    Returns (D, violation), both length len(F).  D is unnormalised: divide by
    v_scale for the reported quantity.
    """
    M, d = U.shape
    A = np.empty((d + 1, M))
    A[:d, :] = U.T
    A[d, :] = lam
    b = np.empty(d + 1)
    b[d] = lam

    D = np.empty(len(F))
    viol = np.empty(len(F))
    for q, f in enumerate(F):
        b[:d] = f
        alpha, _ = nnls(A, b, maxiter=maxiter_mult * M)
        D[q] = np.linalg.norm(f - U.T @ alpha)
        viol[q] = abs(alpha.sum() - 1.0)
    return D, viol


def check_simplex(viol, tol=SIMPLEX_TOL):
    worst = float(np.max(viol)) if len(viol) else 0.0
    if worst >= tol:
        raise AssertionError(
            f"simplex constraint violated: max |sum alpha - 1| = {worst:.2e} "
            f">= {tol:.0e}; raise LAMBDA")
    return worst


def coverage_proxy(d_nn, d, V):
    """rho(x) = d_nn(x) / sqrt(d V): nearest-neighbour distance in units of
    the typical separation between stored states."""
    return d_nn / np.sqrt(d * V)


def lipschitz_estimate(C, U, q=95, chunk=1024):
    """Upper-bound slope of the target map, from each bank point to its
    nearest bank neighbour.

    Reported as the qth percentile so that a handful of near-duplicate stored
    states cannot dominate it.

    Brute force in chunks rather than a KD-tree: the banks here reach
    M = 20000 at d = 64, and spatial trees degrade to full scans above about
    twenty dimensions while paying tree-traversal overhead on top.  A blocked
    matrix product is a few seconds, and the full distance matrix is never
    held at once.
    """
    M = C.shape[0]
    c2 = np.einsum("ij,ij->i", C, C)
    dc = np.empty(M)
    nn = np.empty(M, dtype=np.intp)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        d2 = c2[s:e, None] + c2[None, :] - 2.0 * (C[s:e] @ C.T)
        np.maximum(d2, 0.0, out=d2)
        # Exclude each point as its own neighbour.
        d2[np.arange(e - s), np.arange(s, e)] = np.inf
        j = d2.argmin(1)
        nn[s:e] = j
        dc[s:e] = np.sqrt(d2[np.arange(e - s), j])

    du = np.linalg.norm(U - U[nn], axis=1)
    ok = dc > 0
    if not ok.any():
        return float("nan")
    return float(np.percentile(du[ok] / dc[ok], q))
