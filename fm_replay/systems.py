"""The table of dynamical systems and Gaussian controls.

Every system carries a fixed integer id, which is the first component of the
RNG spawn key, so a system's draws never depend on the order systems appear
in a grid.  Ids are permanent: append, never renumber.

Two kinds live here.  Dynamical systems (L63, L96 at several widths and
couplings) are integrated with RK4.  The Gaussian control (``Gauss-d``) has no
dynamics at all: its bank is drawn iid N(0, I_d).  It exists so that the
dimensional scaling in E1 can be read against a cloud whose separation law is
known exactly, isolating what is attractor geometry from what is dimension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class System:
    id: int
    name: str
    d: int
    rhs: Callable | None
    params: dict = field(default_factory=dict)
    dt: float = 0.01
    stride: int = 5
    ic: Callable | None = None
    kind: str = "ode"          # "ode" or "gauss"

    @property
    def dtau(self):
        """Physical time between consecutive stored states."""
        return self.dt * self.stride


# ----------------------------------------------------------------------
# Vector fields
# ----------------------------------------------------------------------

def l63_rhs(x, p):
    out = np.empty_like(x)
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    out[..., 0] = p["sigma"] * (Y - X)
    out[..., 1] = X * (p["rho"] - Z) - Y
    out[..., 2] = X * Y - p["beta"] * Z
    return out


def l96_rhs(x, p):
    """dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F, cyclic indices."""
    return (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1) - x + p["F"]


# ----------------------------------------------------------------------
# Initial conditions.  Each takes (d, params, rng) and returns x0.
# ----------------------------------------------------------------------

def ic_normal(d, p, rng):
    return rng.standard_normal(d)


def ic_forced_normal(d, p, rng):
    return p.get("F", 0.0) * np.ones(d) + rng.standard_normal(d)


def ic_bump(d, p, rng):
    """Single-site perturbation. Deterministic; selects a particular F=4 wave."""
    x = p.get("F", 0.0) * np.ones(d)
    x[0] += 0.01
    return x


def ic_cosine(d, p, rng):
    """k=1 cosine seed. Deterministic; selects a different F=4 wave."""
    F = p.get("F", 0.0)
    return F * np.ones(d) + 0.5 * np.cos(2 * np.pi * np.arange(d) / d)


# Systems 4 and 5 are deterministic in their initial condition, so per-seed
# variation for them enters only through the burn-in perturbation applied in
# data.make_data.
DETERMINISTIC_IC = {4, 5}

L63_PARAMS = dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0)

_FIXED = {
    "L63": System(0, "L63", 3, l63_rhs, L63_PARAMS, 0.005, 10, ic_normal),
    "L96-16": System(1, "L96-16", 16, l96_rhs, dict(F=8.0), 0.01, 5, ic_forced_normal),
    "L96-64": System(2, "L96-64", 64, l96_rhs, dict(F=8.0), 0.01, 5, ic_forced_normal),
    "L96-64-F5": System(3, "L96-64-F5", 64, l96_rhs, dict(F=5.0), 0.01, 5, ic_forced_normal),
    "L96-64-F4-bump": System(4, "L96-64-F4-bump", 64, l96_rhs, dict(F=4.0), 0.01, 5, ic_bump),
    "L96-64-F4-cos": System(5, "L96-64-F4-cos", 64, l96_rhs, dict(F=4.0), 0.01, 5, ic_cosine),
}

# Onset sweep: ids 10+j over this fixed dimension list.
ONSET_DIMS = [8, 16, 32, 64, 128, 256]

# Gaussian control: ids 20+j over this fixed dimension list.
GAUSS_DIMS = [8, 16, 32, 64, 128, 256, 512]


def onset_system(d):
    """L96 at width d, F=8, id 10+j with j the index into ONSET_DIMS.

    Named with an explicit suffix because the onset block
    overlaps the fixed table at d=16 and d=64: those are the same equations
    but a different id and a shorter training trajectory, so they must not
    share a name, an RNG stream or a cache file.
    """
    j = ONSET_DIMS.index(d)
    return System(10 + j, f"L96-{d}-onset", d, l96_rhs, dict(F=8.0),
                  0.01, 5, ic_forced_normal)


def gauss_system(d):
    """iid N(0, I_d) control bank, id 20+j. No dynamics, no ground truth."""
    j = GAUSS_DIMS.index(d)
    return System(20 + j, f"Gauss-{d}", d, None, {}, kind="gauss")


def get_system(name):
    if name in _FIXED:
        return _FIXED[name]
    if name.startswith("Gauss-"):
        return gauss_system(int(name.split("-")[1]))
    if name.endswith("-onset"):
        return onset_system(int(name.split("-")[1]))
    raise KeyError(f"unknown system {name!r}")


def all_systems():
    out = dict(_FIXED)
    for d in ONSET_DIMS:
        s = onset_system(d)
        out.setdefault(s.name, s)
    for d in GAUSS_DIMS:
        s = gauss_system(d)
        out[s.name] = s
    return out
