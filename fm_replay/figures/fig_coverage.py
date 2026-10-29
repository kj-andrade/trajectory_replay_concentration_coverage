"""Fig. 3 of the paper: target coverage and the attainable error floor (E2)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..metrics import across_seeds
from ..style import use_style, save, FULL, C, CYCLE, panel_label, legend

use_style()

HATCH = {"heldout": "", "isotropic-noise": "///", "off-0.25": "...",
         "off-1.0": "xxx", "midpoint": "\\\\\\"}
FAMILY_LABEL = {"heldout": "held-out", "isotropic-noise": "training law",
                "off-0.25": "off 0.25", "off-1.0": "off 1.0",
                "midpoint": "midpoint"}


def _cells(rows):
    cell = [r for r in rows if r.get("block") == "cell"]
    keys = sorted({(r["system"], int(r["M"]), r["family"]) for r in cell})
    out = {}
    for k in keys:
        sel = [r for r in cell
               if (r["system"], int(r["M"]), r["family"]) == k]
        out[k] = {f: across_seeds([r[f] for r in sel])
                  for f in ("rho", "D_t", "e_best", "e_nn", "phi")}
    return out


# Below this, D_t is zero to solver precision: the true velocity lies inside
# the convex hull of the stored targets and there is no representation
# obstruction at all.  Measured exact-zero cells sit near 1e-12, so this
# floor separates them from anything meaningful without hiding a real value.
DT_FLOOR = 1e-3


def _panel_ladder(ax, cells, sys_colour):
    """D_t, e_best and e_NN stacked at the coverage of each cell."""
    any_zero = False
    for (system, M, family), a in cells.items():
        x = a["rho"]["med"]
        col = sys_colour[system]
        zero = a["D_t"]["med"] < DT_FLOOR
        any_zero |= zero
        dt = DT_FLOOR if zero else a["D_t"]["med"]

        lo = min(dt, a["e_best"]["med"], a["e_nn"]["med"])
        hi = max(dt, a["e_best"]["med"], a["e_nn"]["med"])
        ax.plot([x, x], [lo, hi], color=col, lw=0.5, alpha=0.55, zorder=2)

        # An exact zero is pinned to the floor and drawn as a downward
        # triangle, so it reads as "at or below" rather than as a value.
        ax.plot(x, dt, marker="v" if zero else "o", markersize=3.2,
                markerfacecolor="white", markeredgecolor=col,
                markeredgewidth=0.6, ls="none", zorder=4)
        if not zero:
            v = a["D_t"]
            ax.errorbar(x, v["med"],
                        yerr=[[max(v["med"] - v["lo"], 0)],
                              [max(v["hi"] - v["med"], 0)]],
                        color=col, lw=0, elinewidth=0.6, capsize=1.2, zorder=3)

        for key, marker in (("e_best", "o"), ("e_nn", "^")):
            v = a[key]
            ax.errorbar(x, v["med"],
                        yerr=[[max(v["med"] - v["lo"], 0)],
                              [max(v["hi"] - v["med"], 0)]],
                        color=col, marker=marker, markersize=3.2,
                        markerfacecolor=col, markeredgecolor=col,
                        markeredgewidth=0.6, lw=0, elinewidth=0.6,
                        capsize=1.2, zorder=4)

    ax.axhline(1.0, color=C["grey"], ls=":", lw=0.75, zorder=1)
    ax.text(0.98, 1.0, "zero-motion error ",
            transform=ax.get_yaxis_transform(), fontsize=5.2,
            color=C["grey"], ha="right", va="bottom")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(bottom=DT_FLOOR / 2.2)
    ax.set_xlabel(r"coverage $\rho=d_{\rm nn}/\sqrt{d\hat V}$")
    ax.set_ylabel("normalised error")

    handles = [Line2D([0], [0], marker="o", ls="none", markerfacecolor="white",
                      markeredgecolor=C["black"], markersize=3.4,
                      label=r"$D_t$ (hull floor)"),
               Line2D([0], [0], marker="o", ls="none", color=C["black"],
                      markersize=3.4, label=r"$e_{\rm best}$"),
               Line2D([0], [0], marker="^", ls="none", color=C["black"],
                      markersize=3.4, label=r"$e_{\rm NN}$")]
    if any_zero:
        handles.append(
            Line2D([0], [0], marker="v", ls="none", markerfacecolor="white",
                   markeredgecolor=C["black"], markersize=3.4,
                   label=r"$D_t=0$: $f$ inside hull"))
    handles += [Line2D([0], [0], marker="s", ls="none", color=c,
                       markersize=3.4, label=s)
                for s, c in sys_colour.items()]
    legend(ax, handles=handles, loc="upper left", fontsize=4.5,
           handlelength=1.0, handletextpad=0.35, labelspacing=0.18,
           borderpad=0.28, borderaxespad=0.22, ncol=2)
    panel_label(ax, "(a)", x=-0.20, y=1.02)


def _panel_phi(ax, cells, sys_colour):
    """What fraction of the attainable error is representation floor."""
    groups = sorted({(s, M) for (s, M, _) in cells})
    families = list(dict.fromkeys(f for (_, _, f) in cells))
    w = 0.8 / max(len(families), 1)

    for gi, (system, M) in enumerate(groups):
        for fi, fam in enumerate(families):
            a = cells.get((system, M, fam))
            if a is None:
                continue
            x = gi + (fi - (len(families) - 1) / 2) * w
            v = a["phi"]
            ax.bar(x, v["med"], w * 0.92, color=sys_colour[system],
                   edgecolor=C["black"], linewidth=0.3,
                   hatch=HATCH.get(fam, ""), zorder=3)
            ax.errorbar(x, v["med"],
                        yerr=[[max(v["med"] - v["lo"], 0)],
                              [max(v["hi"] - v["med"], 0)]],
                        color=C["black"], lw=0, elinewidth=0.5, capsize=1.0,
                        zorder=4)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{s}\n$M{{=}}{M}$" for s, M in groups], fontsize=5.0)
    ax.set_ylabel(r"$\phi=\mathrm{med}\,D_t\,/\,\mathrm{med}\,e_{\rm best}$")
    ax.set_ylim(0, min(1.05, max(
        a["phi"]["hi"] for a in cells.values() if np.isfinite(a["phi"]["hi"]))
        * 1.15 if cells else 1.05))
    ax.grid(axis="x", visible=False)

    handles = [Patch(facecolor="white", edgecolor=C["black"], linewidth=0.3,
                     hatch=HATCH.get(f, ""), label=FAMILY_LABEL.get(f, f))
               for f in families]
    legend(ax, handles=handles, loc="upper right", fontsize=4.6,
           handlelength=1.0, handletextpad=0.35, labelspacing=0.2,
           borderpad=0.3, borderaxespad=0.25, ncol=2)
    panel_label(ax, "(b)", x=-0.16, y=1.02)


def figure_coverage(rows, outdir, stem="fig3_coverage"):
    cells = _cells(rows)
    systems = list(dict.fromkeys(s for (s, _, _) in cells))
    sys_colour = {s: CYCLE[i % len(CYCLE)] for i, s in enumerate(systems)}

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.6))
    _panel_ladder(a, cells, sys_colour)
    _panel_phi(b, cells, sys_colour)
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.235, top=0.955,
                        wspace=0.30)
    return save(fig, outdir, stem, tight=False)
