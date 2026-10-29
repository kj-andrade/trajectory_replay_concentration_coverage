"""Fig. 2 of the paper: replay onset and dimensional scaling (E1)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator

from ..metrics import across_seeds, linear_fit
from ..style import use_style, save, FULL, C, CYCLE, panel_label, legend

use_style()

KI_MARKER = {(1, "linear"): "o", (2, "linear"): "s", (2, "cubic"): "^",
             (4, "linear"): "D", (4, "cubic"): "v"}
GAUSS_COLOUR = C["black"]

# Panels (a) and (b) are read at a flow time that is not a knot of any
# window order, since at a knot every interpolant coincides exactly and
# the interpolant comparison would be vacuous.
T_PREFERRED = 0.35


def _pick_t(rows):
    ts = sorted({float(r["t"]) for r in rows if r.get("t") is not None})
    if not ts:
        return None
    return min(ts, key=lambda t: abs(t - T_PREFERRED))


def _at_t(rows, t):
    if t is None:
        return rows
    return [r for r in rows if r.get("t") is not None
            and np.isclose(float(r["t"]), t)]


def _by(rows, **sel):
    out = rows
    for k, v in sel.items():
        out = [r for r in out if r.get(k) == v]
    return out


def _curve(rows, xkey, ykey):
    """Median across seeds with a min-max band, at each x."""
    xs = sorted({float(r[xkey]) for r in rows if r.get(xkey) is not None})
    med, lo, hi = [], [], []
    for x in xs:
        a = across_seeds([r[ykey] for r in rows
                          if np.isclose(float(r[xkey]), x)])
        med.append(a["med"]); lo.append(a["lo"]); hi.append(a["hi"])
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)


def _ki(r):
    return (int(r["k"]), r["interpolant"]) if r["k"] != "" else None


# ---------------------------------------------------------------------
# (a) the ESS master curve
# ---------------------------------------------------------------------

def _panel_ess(ax, rows):
    sweep = _by(rows, block="sweep")
    t = _pick_t(sweep)
    sweep = _at_t(sweep, t)
    if not sweep:
        return
    M = int(sweep[0]["M"])

    dims = sorted({int(r["d"]) for r in sweep if r["kind"] == "l96"})
    colour = {d: CYCLE[i % len(CYCLE)] for i, d in enumerate(dims)}

    for d in dims:
        for ki in sorted({_ki(r) for r in _by(sweep, kind="l96", d=d)}):
            sel = [r for r in _by(sweep, kind="l96", d=d) if _ki(r) == ki]
            x, m, lo, hi = _curve(sel, "kappa", "ess")
            ax.fill_between(x, lo, hi, color=colour[d], alpha=0.10, lw=0)
            ax.plot(x, m, color=colour[d], lw=0.8, alpha=0.9,
                    marker=KI_MARKER.get(ki, "o"), markersize=2.4,
                    markevery=6, zorder=3)

    # The control: an iid Gaussian cloud, whose separation law is exact.
    for d in sorted({int(r["d"]) for r in _by(sweep, kind="gauss")}):
        sel = _by(sweep, kind="gauss", d=d)
        x, m, _, _ = _curve(sel, "kappa", "ess")
        ax.plot(x, m, color=GAUSS_COLOUR, lw=0.7, ls=(0, (2, 1.4)),
                alpha=0.55, zorder=2)

    ax.axvline(2 * np.log(M), color=C["grey"], lw=0.7, ls=":", zorder=1)
    ax.text(2 * np.log(M), 0.97, r"  $2\log M$", transform=ax.get_xaxis_transform(),
            fontsize=5.4, color=C["grey"], ha="left", va="top")
    ax.axhline(1.0, color=C["grey"], lw=0.6, ls=":", zorder=1)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"concentration $\hat\kappa=d\hat V_k/\sigma^2$")
    ax.set_ylabel(r"median $\mathrm{ESS}$")

    handles = [Line2D([0], [0], color=colour[d], lw=1.0, label=f"$d={d}$")
               for d in dims]
    handles.append(Line2D([0], [0], color=GAUSS_COLOUR, lw=0.8,
                          ls=(0, (2, 1.4)), label="Gaussian control"))
    legend(ax, handles=handles, loc="lower left", fontsize=4.6,
           title=rf"flow time $t={t:g}$", title_fontsize=4.8,
           handlelength=1.3, handletextpad=0.35, labelspacing=0.2,
           borderpad=0.3, borderaxespad=0.25, ncol=2)
    panel_label(ax, "(a)", x=-0.26, y=1.02)


# ---------------------------------------------------------------------
# (b) separation concentration
# ---------------------------------------------------------------------

def _panel_spread(ax, rows):
    allsweep = _by(rows, block="sweep")
    allsweep = _at_t(allsweep, _pick_t(allsweep))
    if not allsweep:
        return
    # The spread is free of sigma, so any one bandwidth carries it.
    k0 = min(float(q["kappa"]) for q in allsweep)
    sweep = [r for r in allsweep if np.isclose(float(r["kappa"]), k0)]
    if not sweep:
        return

    for ki in sorted({_ki(r) for r in sweep if r["kind"] == "l96"}):
        sel = [r for r in sweep if r["kind"] == "l96" and _ki(r) == ki]
        x, m, lo, hi = _curve(sel, "d", "spread")
        ax.fill_between(x, lo, hi, color=C["blue"], alpha=0.10, lw=0)
        ax.plot(x, m, color=C["blue"], lw=0.8, marker=KI_MARKER.get(ki, "o"),
                markersize=3.0, label=f"L96 $k={ki[0]}$, {ki[1]}", zorder=3)

    sel = [r for r in sweep if r["kind"] == "gauss"]
    if sel:
        x, m, lo, hi = _curve(sel, "d", "spread")
        ax.fill_between(x, lo, hi, color=GAUSS_COLOUR, alpha=0.10, lw=0)
        ax.plot(x, m, color=GAUSS_COLOUR, lw=0.9, ls=(0, (2, 1.4)),
                marker="P", markersize=3.0, label="Gaussian control", zorder=4)

    # Reference slope, anchored at the smallest dimension measured.
    xs = np.array(sorted({float(r["d"]) for r in sweep}))
    anchor = across_seeds([r["spread"] for r in sweep
                           if float(r["d"]) == xs[0]])["med"]
    ax.plot(xs, anchor * (xs / xs[0]) ** -0.5, color=C["red"], ls="--",
            lw=0.9, zorder=2, label=r"$\propto d^{-1/2}$")

    ax.set_xscale("log"); ax.set_yscale("log")
    # Label the measured dimensions only; the log minor ticks would other-
    # wise overprint them.
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(v)}" for v in xs], fontsize=6)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"ambient dimension $d$")
    ax.set_ylabel(r"$\mathrm{sd}(\kappa_{ij})/\mathrm{mean}(\kappa_{ij})$")
    legend(ax, loc="lower left", fontsize=4.6, handlelength=1.3,
           handletextpad=0.35, labelspacing=0.2, borderpad=0.3,
           borderaxespad=0.25)
    panel_label(ax, "(b)", x=-0.26, y=1.02)


# ---------------------------------------------------------------------
# (c) the onset against bank size
# ---------------------------------------------------------------------

def _panel_onset(ax, rows):
    onset = _by(rows, block="onset")
    if not onset:
        return
    systems = list(dict.fromkeys(r["system"] for r in onset))

    for i, s in enumerate(systems):
        sel = _by(onset, system=s)
        x, m, lo, hi = _curve(sel, "M", "kappa_star")
        col = CYCLE[i % len(CYCLE)]
        ax.errorbar(x, m, yerr=[m - lo, hi - m], color=col, marker="o",
                    markersize=3.0, lw=0, elinewidth=0.7, capsize=1.4,
                    zorder=4, label=s)
        # Per-seed fits, aggregated: the line is the median fit.
        cs = [r["slope_c"] for r in _by(rows, block="onset_fit", system=s)]
        bs = [r["intercept_b"] for r in _by(rows, block="onset_fit", system=s)]
        if cs:
            c, b = float(np.median(cs)), float(np.median(bs))
            xx = np.geomspace(x.min(), x.max(), 60)
            ax.plot(xx, c * np.log(xx) + b, color=col, lw=0.9, alpha=0.8,
                    zorder=3)

    Ms = np.geomspace(min(float(r["M"]) for r in onset),
                      max(float(r["M"]) for r in onset), 80)
    for mult, ls, lab in ((2, "--", r"$2\log M$"), (4, ":", r"$4\log M$")):
        ax.plot(Ms, mult * np.log(Ms), color=C["grey"], ls=ls, lw=0.8,
                zorder=2, label=lab)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"bank size $M$")
    ax.set_ylabel(r"onset $\kappa^{\star}$ at $\mathrm{ESS}=2$")
    legend(ax, loc="upper left", fontsize=4.8, handlelength=1.3,
           handletextpad=0.35, labelspacing=0.2, borderpad=0.3,
           borderaxespad=0.25)
    panel_label(ax, "(c)", x=-0.26, y=1.02)


def figure_onset(rows, outdir, stem="fig2_onset"):
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3))
    _panel_ess(axes[0], rows)
    _panel_spread(axes[1], rows)
    _panel_onset(axes[2], rows)
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.215, top=0.955,
                        wspace=0.52)
    return save(fig, outdir, stem, tight=False)
