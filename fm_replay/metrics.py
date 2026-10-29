"""The aggregation rule used for every reported number.

The order matters and is the same everywhere:

  1. per seed, median over test queries;
  2. across seeds, median of those per-seed medians, with a min-max band;
  3. paired quantities (ratios, differences) are formed per seed FIRST and
     only then aggregated, so that a ratio is never a ratio of medians taken
     over different banks.
"""
from __future__ import annotations

import numpy as np


def per_seed(x):
    """Step 1: the per-seed summary of a per-query metric."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def across_seeds(values):
    """Step 2/3: median with a min-max band over seeds."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if not v.size:
        return dict(med=np.nan, lo=np.nan, hi=np.nan, mean=np.nan,
                    sd=np.nan, n=0)
    return dict(med=float(np.median(v)), lo=float(v.min()), hi=float(v.max()),
                mean=float(v.mean()), sd=float(v.std(ddof=1) if v.size > 1 else 0.0),
                n=int(v.size))


def fmt(agg, prec=3):
    """median [min, max], the table format used throughout."""
    if not np.isfinite(agg["med"]):
        return "--"
    return f"{agg['med']:.{prec}f} [{agg['lo']:.{prec}f}, {agg['hi']:.{prec}f}]"


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def loglog_fit(x, y):
    """log y = intercept + slope log x; returns (slope, intercept, r2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return np.nan, np.nan, np.nan
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (intercept + slope * lx)
    sst = float(((ly - ly.mean()) ** 2).sum())
    r2 = np.nan if sst <= 0 else 1.0 - float((resid ** 2).sum()) / sst
    return float(slope), float(intercept), float(r2)


def linear_fit(x, y):
    """y = c x + b; returns (c, b, r2).  Used for kappa* against log M."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    c, b = np.polyfit(x, y, 1)
    resid = y - (c * x + b)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = np.nan if sst <= 0 else 1.0 - float((resid ** 2).sum()) / sst
    return float(c), float(b), float(r2)


def crossing(xs, ys, level):
    """The x at which y first falls to ``level``, interpolated in log-log.

    Returns nan when the curve does not bracket the level, rather than
    extrapolating a crossing the sweep never observed.
    """
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if x.size < 2 or y[0] <= level or y[-1] >= level:
        return float("nan")
    j = int(np.argmax(y < level))
    x0, x1 = np.log(x[j - 1]), np.log(x[j])
    y0, y1 = np.log(y[j - 1]), np.log(y[j])
    if y0 == y1:
        return float(np.exp(x0))
    return float(np.exp(x0 + (np.log(level) - y0) * (x1 - x0) / (y1 - y0)))
