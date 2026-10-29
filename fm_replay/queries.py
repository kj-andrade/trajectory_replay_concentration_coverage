"""Query families for E2 and E3.

Families are drawn once per (system, seed, family) from the query_val or
query_test stream and reused across every M, kappa and geometry, so that
comparisons between operating points are paired on identical queries.

The families walk a query away from the stored support in a controlled way:
``heldout`` sits on the attractor, the two ``off`` families sit at a fixed
fraction of the typical separation off it, ``midpoint`` sits between two
stored branches, and ``isotropic-noise`` is the flow-matching training law,
which in high dimension is NOT an on-support family.
"""
from __future__ import annotations

import numpy as np

from .data import standardize
from .rng import rng as make_rng, QUERY_VAL, QUERY_TEST

FAMILIES = ["heldout", "isotropic-noise", "off-0.25", "off-1.0", "midpoint"]


def _heldout(D, n, r):
    T = D["test"]
    idx = r.choice(len(T) - 1, min(n, len(T) - 1), replace=False)
    return standardize(T[idx], D)


def _unit_directions(shape, r):
    u = r.standard_normal(shape)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def build_family(name, D, C, scale, n, r):
    """One query family in standardised coordinates.

    ``scale`` is sqrt(d V), the typical separation between stored states.
    """
    if name == "heldout":
        return _heldout(D, n, r)
    if name == "isotropic-noise":
        i = r.integers(len(C), size=n)
        return C[i] + r.standard_normal((n, C.shape[1]))
    if name.startswith("off-"):
        radius = float(name.split("-")[1]) * scale
        X = _heldout(D, n, r)
        return X + radius * _unit_directions(X.shape, r)
    if name == "midpoint":
        i = r.integers(len(C), size=n)
        j = r.integers(len(C), size=n)
        return 0.5 * (C[i] + C[j])
    raise ValueError(f"unknown query family {name!r}")


def build_families(D, C, scale, n, split, names=None):
    """All families for one split, ``split`` in {'val', 'test'}.

    Each family gets its own generator, addressed by its fixed index in
    FAMILIES, so a family drawn for E3 (which omits the training law) is
    identical to the same family drawn for E2.
    """
    stream = QUERY_VAL if split == "val" else QUERY_TEST
    S = D["system"]
    out = {}
    for name in (names or FAMILIES):
        r = make_rng(S.id, D["seed"], stream, sub=(FAMILIES.index(name),))
        out[name] = build_family(name, D, C, scale, n, r)
    return out
