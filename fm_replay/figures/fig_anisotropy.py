"""Fig. 4 of the paper: covariance geometry at matched concentration (E3)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..experiments.anisotropy import RATIO_GEOM
from ..metrics import across_seeds
from ..style import use_style, save, FULL, C, CYCLE, panel_label, legend

use_style()

FAM_MARKER = {"heldout": "o", "off-0.25": "s", "off-1.0": "^",
              "midpoint": "D", "isotropic-noise": "v"}
ARM_STYLE = {"iso": (C["blue"], "o", "isotropic"),
             "halfW-trace": (C["red"], "s",
                             r"$\Sigma\propto\Gamma^{1/2}$, equal trace")}


def _by(rows, **sel):
    out = rows
    for k, v in sel.items():
        out = [r for r in out if r.get(k) == v]
    return out


def _panel_ratio(ax, rows, tol, sys_colour):
    """Shape at matched concentration: does the ratio sit inside the band?"""
    ratio = _by(rows, block="ratio", geometry=RATIO_GEOM)
    if not ratio:
        return 0, 0

    ax.axhspan(1 - tol, 1 + tol, color=C["light"], alpha=0.30, lw=0, zorder=0)
    ax.axhline(1.0, color=C["black"], lw=0.75, zorder=1)

    matched = total = 0
    for system in sorted({r["system"] for r in ratio}):
        for fam in sorted({r["family"] for r in ratio}):
            sel = _by(ratio, system=system, family=fam)
            for kappa in sorted({float(r["kappa_target"]) for r in sel}):
                a = across_seeds([r["R"] for r in sel
                                  if np.isclose(float(r["kappa_target"]), kappa)])
                if not np.isfinite(a["med"]):
                    continue
                total += 1
                # A cell counts as matched only if the WHOLE seed range is
                # inside the band declared before the run.
                inside = (a["lo"] >= 1 - tol) and (a["hi"] <= 1 + tol)
                matched += int(inside)
                ax.errorbar(kappa, a["med"],
                            yerr=[[max(a["med"] - a["lo"], 0)],
                                  [max(a["hi"] - a["med"], 0)]],
                            color=sys_colour[system],
                            marker=FAM_MARKER.get(fam, "o"), markersize=2.8,
                            lw=0, elinewidth=0.5, capsize=1.0, alpha=0.85,
                            zorder=3)

    ax.text(0.97, 0.97,
            f"inside $\\pm{100*tol:.0f}\\%$: {matched}/{total}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.4,
            color=C["grey"])

    ax.set_xscale("log")
    ax.set_xlabel(r"matched $\bar\kappa$")
    ax.set_ylabel(r"error ratio, shape / isotropic")

    handles = [Line2D([0], [0], marker="s", ls="none", color=c, markersize=3.2,
                      label=s) for s, c in sys_colour.items()]
    handles += [Line2D([0], [0], marker=FAM_MARKER.get(f, "o"), ls="none",
                       markerfacecolor="white", markeredgecolor=C["black"],
                       markeredgewidth=0.6, markersize=3.2, label=f)
                for f in sorted({r["family"] for r in ratio})]
    legend(ax, handles=handles, loc="lower left", fontsize=4.4,
           handlelength=1.0, handletextpad=0.3, labelspacing=0.18,
           borderpad=0.28, borderaxespad=0.22, ncol=2)
    panel_label(ax, "(a)", x=-0.22, y=1.02)
    return matched, total


def _panel_equal_trace(ax, rows):
    """What equal-trace half-whitening trades, on the system that should
    benefit most."""
    sysrows = _by(rows, block="system")
    if not sysrows:
        return
    gate = {}
    for r in sysrows:
        gate.setdefault(r["system"], float(r["reff_over_d"]))
    target = min(gate, key=gate.get)

    fam = "heldout"
    arms = _by(rows, block="arm", system=target, family=fam)
    for g, (col, mk, lab) in ARM_STYLE.items():
        sel = _by(arms, geometry=g)
        xs = sorted({float(r["Ekappa"]) for r in sel})
        med, lo, hi = [], [], []
        for x in xs:
            a = across_seeds([r["nerr"] for r in sel
                              if np.isclose(float(r["Ekappa"]), x)])
            med.append(a["med"]); lo.append(a["lo"]); hi.append(a["hi"])
        ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(xs, med, color=col, marker=mk, markersize=3.0, lw=1.0,
                label=lab, zorder=3)

    # The decomposition, read at the bandwidth nearest the onset scale.
    eq = _by(rows, block="equal_trace", system=target, family=fam)
    if eq:
        onset = float(eq[0]["onset"])
        k0 = min({float(r["kappa_target"]) for r in eq},
                 key=lambda k: abs(np.log(k) - np.log(onset)))
        sel = [r for r in eq if np.isclose(float(r["kappa_target"]), k0)]
        e1 = across_seeds([r["e1"] for r in sel])["med"]
        e2 = across_seeds([r["e2"] for r in sel])["med"]
        e3 = across_seeds([r["e3"] for r in sel])["med"]
        psi = across_seeds([r["psi"] for r in sel])["med"]
        k_real = float(sel[0]["kappa_realised"])

        # e2 and e3 land at the same concentration and, when the shape
        # refunds nothing, at nearly the same error, so their labels are
        # pushed apart deliberately.
        for x, y, lab, col, off in ((k0, e1, "$e_1$", C["blue"], (5, -1)),
                                    (k_real, e2, "$e_2$", C["red"], (6, -9)),
                                    (k_real, e3, "$e_3$", C["blue"], (6, 5))):
            ax.plot([x], [y], marker="o", markersize=5.0,
                    markerfacecolor="none", markeredgecolor=col,
                    markeredgewidth=0.9, zorder=5)
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=off, fontsize=5.4, color=col, zorder=6)

        ax.text(0.03, 0.03,
                "\n".join((
                    f"{target}",
                    rf"$r_{{\rm eff}}/d={gate[target]:.3f}$",
                    rf"$e_2/e_1={e2/e1:.1f}\times$ at equal trace",
                    rf"$\psi={psi:.2f}$ refunded by shape",
                )),
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=5.0, color=C["grey"], linespacing=1.4)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"realised $\bar\kappa^{\Sigma}=\mathrm{tr}(\Sigma^{-1}\hat\Gamma)$")
    ax.set_ylabel("normalised error")
    legend(ax, loc="upper right", fontsize=4.8, handlelength=1.3,
           handletextpad=0.35, labelspacing=0.2, borderpad=0.3,
           borderaxespad=0.25)
    panel_label(ax, "(b)", x=-0.22, y=1.02)


def figure_anisotropy(rows, outdir, stem="fig4_anisotropy", tol=0.10):
    systems = list(dict.fromkeys(r["system"] for r in rows
                                 if r.get("block") == "system"))
    sys_colour = {s: CYCLE[i % len(CYCLE)] for i, s in enumerate(systems)}

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.6))
    _panel_ratio(a, rows, tol, sys_colour)
    _panel_equal_trace(b, rows)
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.185, top=0.955,
                        wspace=0.32)
    return save(fig, outdir, stem, tight=False)
