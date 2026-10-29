"""Figures 5-7 of the paper.

Every panel renders from partial results: a cell with no rows yet is simply
absent from its curve, and a figure with no rows at all is skipped rather
than raised.

Aggregation follows the rest of the study: median over queries within a run, then the median across
seeds with the seed min-max as the error bar.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from . import paths
from fm_replay.metrics import across_seeds
from fm_replay.style import (use_style, save, FULL, C, CYCLE, panel_label,
                             legend, legend_above)

use_style()

W_COLOUR = {128: C["blue"], 512: C["orange"]}
STAR_COLOUR = C["black"]
FLOOR_COLOUR = C["grey"]
T_MAIN = 0.5


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------

def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return v


def load(results=None, runs=None):
    """Rows from metrics_all.csv, or straight from the run directories."""
    results = Path(results or paths.RESULTS) / "metrics_all.csv"
    if results.exists():
        with results.open(newline="") as f:
            return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]
    rows = []
    for p in sorted(Path(runs or paths.RUNS).glob("*/metrics.csv")):
        with p.open(newline="") as f:
            rows += [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return rows


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        if v is None:
            continue
        out = [r for r in out if (r.get(k) == v
                                  or (isinstance(v, float)
                                      and isinstance(r.get(k), float)
                                      and np.isclose(r[k], v)))]
    return out


def final_step(rows):
    steps = [r["step"] for r in rows if r.get("step") is not None]
    return max(steps) if steps else None


def curve(rows, xkey="lambda"):
    """(x, median, lo, hi) across seeds, sorted by x."""
    xs = sorted({r[xkey] for r in rows if r.get(xkey) is not None})
    m, lo, hi = [], [], []
    for x in xs:
        a = across_seeds([r["median"] for r in rows
                          if np.isclose(r[xkey], x)])
        m.append(a["med"]); lo.append(a["lo"]); hi.append(a["hi"])
    return (np.array(xs), np.array(m), np.array(lo), np.array(hi))


def _errbar(ax, x, m, lo, hi, **kw):
    ok = np.isfinite(m)
    if not ok.any():
        return
    ax.errorbar(x[ok], m[ok],
                yerr=[np.maximum(m[ok] - lo[ok], 0), np.maximum(hi[ok] - m[ok], 0)],
                elinewidth=0.6, capsize=1.4, **kw)


def _onset_marker(ax, rows):
    """kappa_hat = lambda log M, so the 2 log M onset sits at lambda = 2."""
    ax.axvline(2.0, color=C["grey"], lw=0.7, ls=":", zorder=1)
    ax.text(2.0, 0.985, r"  $2\log M$", transform=ax.get_xaxis_transform(),
            fontsize=5.0, color=C["grey"], ha="left", va="top")


def _plain(v, _pos):
    """0.1, 0.2, 2, 5 -- not 2.65 x 10^0, which is unreadable at 5 pt."""
    return "" if v <= 0 else f"{v:g}"


def _tidy_log_y(ax, values):
    """Pick a y scale that actually prints ticks inside the panel.

    A span under a factor of three gets a linear axis; a span of one or two
    decades gets a log axis with decade, half-decade and third-decade ticks
    labelled in plain decimals, since a bare decade locator would leave such
    a panel with a single tick; anything wider gets the default log axis.
    """
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0], float)
    if not v.size:
        return
    span = v.max() / v.min()
    if span < 3.0:
        ax.set_yscale("linear")
        return
    ax.set_yscale("log")
    if span < 100.0:
        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_plain))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())


# ----------------------------------------------------------------------
# the three panels
# ----------------------------------------------------------------------

XLABEL = r"$\lambda=\hat\kappa/\log M$"


def _lambda_axis(ax, rows):
    """Tick at the lambda values that were actually run.

    The default log locator labels minor decades when the sweep covers less
    than one decade, and those labels overprint each other in a 1.2 in panel.
    """
    lams = sorted({r["lambda"] for r in rows if r.get("lambda") is not None})
    ax.set_xscale("log")
    if not lams:
        return
    ax.set_xticks(lams)
    ax.set_xticklabels([f"{v:g}" for v in lams], fontsize=6)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def _note(ax, text):
    """The query set, as a small centred title: the panel interior is
    already carrying the 2 log M marker and the curves."""
    ax.set_title(text, fontsize=5.4, color=C["grey"], pad=2.5)


def panel_fidelity(ax, rows, t, widths, label="(a)"):
    vals = []
    for w in widths:
        for setname, ls, mk in (("IN", "-", "o"), ("HO", "--", "s")):
            s = sel(rows, metric="r", set=setname, t=t, W=w)
            if not s:
                continue
            x, m, lo, hi = curve(s)
            vals += [*lo, *hi]
            _errbar(ax, x, m, lo, hi, color=W_COLOUR.get(w, CYCLE[0]),
                    ls=ls, lw=1.0, marker=mk, markersize=2.6)
    ax.axhline(0.1, color=C["grey"], lw=0.6, ls=(0, (1, 2)), zorder=1)
    _onset_marker(ax, rows)
    _lambda_axis(ax, rows)
    _tidy_log_y(ax, vals + [0.1])
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(r"$\|v_\theta-v^\star\|\,/\,u_{\rm rms}$")
    _note(ax, "in-sample (solid) / held-out (dashed)")
    panel_label(ax, label, x=-0.27, y=1.04)


def panel_replay(ax, rows, t, widths, label="(b)"):
    s = sel(rows, metric="rep_star", set="IN", t=t)
    if s:
        x, m, lo, hi = curve(s)
        _errbar(ax, x, m, lo, hi, color=STAR_COLOUR, ls=(0, (2, 1.4)),
                lw=0.9, marker="P", markersize=2.8)
    for w in widths:
        s = sel(rows, metric="rep_theta", set="IN", t=t, W=w)
        if not s:
            continue
        x, m, lo, hi = curve(s)
        _errbar(ax, x, m, lo, hi, color=W_COLOUR.get(w, CYCLE[0]), ls="-",
                lw=1.0, marker="o", markersize=2.6)
    _onset_marker(ax, rows)
    _lambda_axis(ax, rows)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("replay fraction")
    _note(ax, "in-sample")
    panel_label(ax, label, x=-0.27, y=1.04)


def panel_error(ax, rows, t, widths, setname="OFF", label="(c)"):
    vals = []
    s = sel(rows, metric="D_t", set=setname, t=t)
    if s:
        x, m, lo, hi = curve(s)
        vals += [*m]
        ax.plot(x, m, color=FLOOR_COLOUR, lw=1.1, ls="-", zorder=2)
        ax.fill_between(x, lo, hi, color=FLOOR_COLOUR, alpha=0.18, lw=0)
    s = sel(rows, metric="e_star", set=setname, t=t)
    if s:
        x, m, lo, hi = curve(s)
        vals += [*lo, *hi]
        _errbar(ax, x, m, lo, hi, color=STAR_COLOUR, ls=(0, (2, 1.4)),
                lw=0.9, marker="P", markersize=2.8)
    for w in widths:
        s = sel(rows, metric="e_theta", set=setname, t=t, W=w)
        if not s:
            continue
        x, m, lo, hi = curve(s)
        vals += [*lo, *hi]
        _errbar(ax, x, m, lo, hi, color=W_COLOUR.get(w, CYCLE[0]), ls="-",
                lw=1.0, marker="o", markersize=2.6)
    _onset_marker(ax, rows)
    _lambda_axis(ax, rows)
    _tidy_log_y(ax, vals)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("normalised error")
    _note(ax, "off-support" if setname == "OFF" else "held-out")
    panel_label(ax, label, x=-0.27, y=1.04)


def _legend_handles(widths, off_name):
    h = [Line2D([0], [0], color=STAR_COLOUR, ls=(0, (2, 1.4)), lw=0.9,
                marker="P", markersize=2.8, label=r"$v^\star$ closed form")]
    h += [Line2D([0], [0], color=W_COLOUR.get(w, CYCLE[0]), ls="-", lw=1.0,
                 marker="o", markersize=2.6, label=rf"$v_\theta$, $W={w}$")
          for w in widths]
    h += [Line2D([0], [0], color=FLOOR_COLOUR, ls="-", lw=1.1,
                 label=rf"$D_t$ hull floor ({off_name})")]
    return h


def _top_legend(fig, widths, off_name, y=0.995):
    return fig.legend(handles=_legend_handles(widths, off_name),
                      loc="upper center", bbox_to_anchor=(0.54, y),
                      ncol=len(widths) + 2, frameon=False, fontsize=5.4,
                      handlelength=1.7, handletextpad=0.4,
                      columnspacing=1.3, borderaxespad=0.0)


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------

def fig_main(rows, outdir=None, t=T_MAIN, off_name="off-1.0",
             stem="fig5_networks"):
    rows = [r for r in rows if r.get("step") == final_step(rows)]
    if not rows:
        return None
    widths = sorted({int(r["W"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.15))
    panel_fidelity(axes[0], rows, t, widths)
    panel_replay(axes[1], rows, t, widths)
    have_off = bool(sel(rows, metric="e_theta", set="OFF"))
    panel_error(axes[2], rows, t, widths, "OFF" if have_off else "HO")
    _top_legend(fig, widths, off_name)
    fig.text(0.995, 0.965, rf"$t={t:g}$", ha="right", va="top", fontsize=5.4,
             color=C["grey"])
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.215, top=0.825,
                        wspace=0.40)
    return save(fig, outdir or paths.FIGURES, stem, tight=False)


def fig_training(rows, outdir=None, t=T_MAIN, width=512,
                 stem="fig7b_training"):
    s = sel(rows, metric="r", set="IN", t=t, W=float(width))
    if not s:
        return None
    lams = sorted({r["lambda"] for r in s})
    fig, ax = plt.subplots(1, 1, figsize=(FULL * 0.56, 2.15))
    vals = []
    for i, lam in enumerate(lams):
        x, m, lo, hi = curve([r for r in s if np.isclose(r["lambda"], lam)],
                             xkey="step")
        col = CYCLE[i % len(CYCLE)]
        vals += [*lo, *hi]
        ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(x, m, color=col, lw=1.0, marker="o", markersize=2.6,
                label=rf"$\lambda={lam:g}$")
    ax.axhline(0.1, color=C["grey"], lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.set_xscale("log")
    _tidy_log_y(ax, vals + [0.1])
    ax.set_xlabel("training step")
    ax.set_ylabel(r"fidelity $r$ (in-sample)")
    legend(ax, loc="lower left", fontsize=5.2, title=rf"$W={width}$, $t={t:g}$",
           title_fontsize=5.4, handlelength=1.3, handletextpad=0.35,
           labelspacing=0.2, borderpad=0.3, borderaxespad=0.25, ncol=2)
    fig.subplots_adjust(left=0.165, right=0.985, bottom=0.175, top=0.965)
    return save(fig, outdir or paths.FIGURES, stem, tight=False)


def fig_all_t(rows, outdir=None, off_name="off-1.0", stem="fig6_flow_times"):
    rows = [r for r in rows if r.get("step") == final_step(rows)]
    if not rows:
        return None
    widths = sorted({int(r["W"]) for r in rows})
    ts = sorted({r["t"] for r in rows if r.get("t") is not None})
    have_off = bool(sel(rows, metric="e_theta", set="OFF"))
    n = len(ts)
    fig, axes = plt.subplots(n, 3, figsize=(FULL, 1.85 * n))
    axes = np.atleast_2d(axes)
    for i, t in enumerate(ts):
        panel_fidelity(axes[i, 0], rows, t, widths, f"({'abc'[i]}1)")
        panel_replay(axes[i, 1], rows, t, widths, f"({'abc'[i]}2)")
        panel_error(axes[i, 2], rows, t, widths,
                    "OFF" if have_off else "HO", f"({'abc'[i]}3)")
        axes[i, 0].text(-0.52, 0.5, rf"$t={t:g}$",
                        transform=axes[i, 0].transAxes, rotation=90,
                        va="center", ha="center", fontsize=8)
    _top_legend(fig, widths, off_name, y=0.998)
    top = 1.0 - 0.30 / (1.85 * n)
    fig.subplots_adjust(left=0.135, right=0.995, bottom=0.075 / n * 2,
                        top=top, wspace=0.40, hspace=0.50)
    return save(fig, outdir or paths.FIGURES, stem, tight=False)


def _panel_floor(ax, rows, t, widths, suffix, title, label):
    """Held-out error against one choice of reference field."""
    vals = []
    for metric, col, ls, mk in ((f"D_t{suffix}", FLOOR_COLOUR, "-", None),
                                (f"e_nn{suffix}", C["green"], "-.", "^"),
                                (f"e_star{suffix}", STAR_COLOUR, (0, (2, 1.4)), "P")):
        s = sel(rows, metric=metric, set="HO", t=t)
        if not s:
            continue
        x, m, lo, hi = curve(s)
        vals += [*lo, *hi]
        if mk is None:
            ax.plot(x, m, color=col, lw=1.1, ls=ls, zorder=2)
            ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        else:
            _errbar(ax, x, m, lo, hi, color=col, ls=ls, lw=0.9, marker=mk,
                    markersize=2.8)
    for w in widths:
        s = sel(rows, metric=f"e_theta{suffix}", set="HO", t=t, W=w)
        if not s:
            continue
        x, m, lo, hi = curve(s)
        vals += [*lo, *hi]
        _errbar(ax, x, m, lo, hi, color=W_COLOUR.get(w, CYCLE[0]), ls="-",
                lw=1.0, marker="o", markersize=2.6)
    _onset_marker(ax, rows)
    _lambda_axis(ax, rows)
    _tidy_log_y(ax, vals)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("normalised error")
    _note(ax, title)
    panel_label(ax, label, x=-0.155, y=1.04)


def fig_floor(rows, outdir=None, t=T_MAIN, stem="fig7a_reference"):
    """Fig. 7a: the escape question under two reference fields.

    Left, the reference is the true field at the query point, so the floor
    moves with lambda.  Right, it is the held-out window's own displacement,
    so the floor is fixed and the two curves can be read against each other.
    """
    rows = [r for r in rows if r.get("step") == final_step(rows)]
    if not rows or not sel(rows, metric="D_t_u", set="HO"):
        return None
    widths = sorted({int(r["W"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.25))
    _panel_floor(axes[0], rows, t, widths, "",
                 r"held-out, reference $f_t(x)$ at the query", "(a)")
    _panel_floor(axes[1], rows, t, widths, "_u",
                 r"held-out, reference $u^\star$ of the window", "(b)")
    h = [Line2D([0], [0], color=STAR_COLOUR, ls=(0, (2, 1.4)), lw=0.9,
                marker="P", markersize=2.8, label=r"$v^\star$"),
         Line2D([0], [0], color=C["green"], ls="-.", lw=0.9, marker="^",
                markersize=2.8, label=r"$u_{i^\star}$ (1-NN)")]
    h += [Line2D([0], [0], color=W_COLOUR.get(w, CYCLE[0]), ls="-", lw=1.0,
                 marker="o", markersize=2.6, label=rf"$v_\theta$, $W={w}$")
          for w in widths]
    h += [Line2D([0], [0], color=FLOOR_COLOUR, lw=1.1, label=r"$D_t$ floor")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.54, 0.995),
               ncol=len(h), frameon=False, fontsize=5.4, handlelength=1.7,
               handletextpad=0.4, columnspacing=1.3, borderaxespad=0.0)
    fig.text(0.995, 0.955, rf"$t={t:g}$", ha="right", va="top", fontsize=5.4,
             color=C["grey"])
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.205, top=0.835,
                        wspace=0.24)
    return save(fig, outdir or paths.FIGURES, stem, tight=False)


def make_all(rows=None, outdir=None, off_name="off-1.0"):
    rows = load() if rows is None else rows
    if not rows:
        print("no rows yet")
        return []
    out = []
    for f in (fig_main(rows, outdir, off_name=off_name),
              fig_training(rows, outdir),
              fig_all_t(rows, outdir, off_name=off_name),
              fig_floor(rows, outdir)):
        if f:
            out.append(f)
            print(f"  wrote {f[0].name}, {f[1].name}")
    return out
