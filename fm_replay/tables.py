"""The reported numbers, as Markdown tables.

Every entry follows the aggregation rule in metrics.py: a per-seed summary
first, then median [min, max] across seeds.
"""
from __future__ import annotations

import numpy as np

from .metrics import across_seeds, fmt, spearman


def _rows(rows, **sel):
    out = rows
    for k, v in sel.items():
        out = [r for r in out if r.get(k) == v]
    return out


def _table(header, body):
    if not body:
        return ""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- E1
def table1(rows):
    """V_k per cell, the onset slope c, and the gap-law KS p-values."""
    sweep = _rows(rows, block="sweep")
    body = []
    seen = set()
    for r in sweep:
        key = (r["system"], int(r["d"]), r["k"], r["interpolant"])
        if key in seen:
            continue
        seen.add(key)
        sel = [q for q in sweep if (q["system"], int(q["d"]), q["k"],
                                    q["interpolant"]) == key]
        vk = across_seeds([q["Vk"] for q in sel])
        sp = across_seeds([q["spread"] for q in sel])
        body.append([key[0], key[1], key[2], key[3], fmt(vk), fmt(sp, 4)])
    out = ["**Table 1a — separation constant and separation spread**\n",
           _table(["system", "d", "k", "interpolant",
                   "V_k  median [min, max]", "sd/mean"], body)]

    ks = _rows(rows, block="ks")
    body = []
    for s in sorted({r["system"] for r in ks}):
        p = across_seeds([r["ks_pvalue"] for r in _rows(ks, system=s)])
        body.append([s, f"{p['med']:.3f}", f"{p['lo']:.3f}", f"{p['hi']:.3f}"])
    out += ["\n**Table 1b — gap-law KS test against N(0,1)** "
            "(implementation check, not evidence about the theory)\n",
            _table(["system", "p median", "p min", "p max"], body)]

    fits = _rows(rows, block="onset_fit")
    body = []
    for s in sorted({r["system"] for r in fits}):
        sel = _rows(fits, system=s)
        c = across_seeds([r["slope_c"] for r in sel])
        b = across_seeds([r["intercept_b"] for r in sel])
        r2 = across_seeds([r["r2"] for r in sel])
        body.append([s, fmt(c, 2), fmt(b, 2), fmt(r2, 3)])
    out += ["\n**Table 1c — onset fit  kappa* = c log M + b**\n",
            _table(["system", "c", "b", "R^2"], body)]
    return "\n".join(out)


# ----------------------------------------------------------------- E2
def table2(rows):
    cell = _rows(rows, block="cell")
    body = []
    for key in sorted({(r["system"], int(r["M"]), r["family"]) for r in cell}):
        sel = [r for r in cell
               if (r["system"], int(r["M"]), r["family"]) == key]
        g = {f: across_seeds([r[f] for r in sel])
             for f in ("D_t", "e_best", "e_nn", "lipschitz_bound", "phi",
                       "rho", "kappa_star")}
        flag = "yes" if any(r["kappa_star_at_boundary"] in (True, "True")
                            for r in sel) else ""
        body.append([key[0], key[1], key[2], fmt(g["rho"], 3),
                     fmt(g["D_t"], 3), fmt(g["e_best"], 3), fmt(g["e_nn"], 3),
                     fmt(g["lipschitz_bound"], 2), fmt(g["phi"], 3),
                     f"{g['kappa_star']['med']:.1f}", flag])
    out = ["**Table 2 — coverage, hull floor and attainable error**\n",
           _table(["system", "M", "family", "rho", "D_t", "e_best", "e_NN",
                   "L d_nn", "phi", "kappa*", "at grid edge"], body)]

    # Spearman between coverage and the attainable floor, per seed.
    rhos = []
    for seed in sorted({r["seed"] for r in cell}):
        sel = _rows(cell, seed=seed)
        rhos.append(spearman([r["rho"] for r in sel],
                             [r["e_best"] for r in sel]))
    a = across_seeds(rhos)
    out.append(f"\nSpearman(rho, e_best) across cells, per seed: "
               f"{fmt(a, 3)}\n")
    return "\n".join(out)


# ----------------------------------------------------------------- E3
def table3(rows):
    sysrows = _rows(rows, block="system")
    body = []
    for s in sorted({r["system"] for r in sysrows}):
        sel = _rows(sysrows, system=s)
        mc = [k for k in sel[0] if str(k).startswith("mc_check_")]
        checks = ", ".join(
            f"{k.replace('mc_check_', '')}={across_seeds([r[k] for r in sel])['med']:.3f}"
            for k in sorted(mc))
        body.append([s, int(sel[0]["d"]),
                     fmt(across_seeds([r["V"] for r in sel]), 3),
                     fmt(across_seeds([r["reff"] for r in sel]), 2),
                     fmt(across_seeds([r["reff_over_d"] for r in sel]), 4),
                     checks])
    return ("**Table 3 — pair-displacement spectrum**\n\n"
            + _table(["system", "d", "V", "r_eff", "r_eff/d",
                      "MC check of kappa_bar (ratio to closed form)"], body))


def table4(rows, tol=0.10):
    ratio = _rows(rows, block="ratio")
    body = []
    overall_m = overall_n = 0
    for fam in sorted({r["family"] for r in ratio}):
        m = n = 0
        for key in sorted({(r["system"], r["geometry"],
                            float(r["kappa_target"]))
                           for r in _rows(ratio, family=fam)}):
            sel = [r for r in _rows(ratio, family=fam)
                   if (r["system"], r["geometry"],
                       float(r["kappa_target"])) == key]
            a = across_seeds([r["R"] for r in sel])
            if not np.isfinite(a["med"]):
                continue
            n += 1
            m += int(a["lo"] >= 1 - tol and a["hi"] <= 1 + tol)
        body.append([fam, f"{m}/{n}", f"{100*m/n:.0f}%" if n else "--"])
        overall_m += m; overall_n += n
    body.append(["**all**", f"**{overall_m}/{overall_n}**",
                 f"**{100*overall_m/overall_n:.0f}%**" if overall_n else "--"])
    out = [f"**Table 4a — matched-concentration test, declared tol = "
           f"{tol:.2f}**\n",
           _table(["family", "matched k/N", "fraction"], body)]

    eq = _rows(rows, block="equal_trace", family="heldout")
    body = []
    for s in sorted({r["system"] for r in eq}):
        for k0 in sorted({float(r["kappa_target"]) for r in _rows(eq, system=s)}):
            sel = [r for r in _rows(eq, system=s)
                   if np.isclose(float(r["kappa_target"]), k0)]
            body.append([s, f"{k0:.1f}",
                         f"{float(sel[0]['kappa_realised']):.2f}",
                         fmt(across_seeds([r["e1"] for r in sel]), 4),
                         fmt(across_seeds([r["e2"] for r in sel]), 4),
                         fmt(across_seeds([r["e3"] for r in sel]), 4),
                         fmt(across_seeds([r["psi"] for r in sel]), 3)])
    out += ["\n**Table 4b — equal-trace decomposition, held-out queries**\n",
            _table(["system", "kappa_0", "kappa realised", "e1 (iso)",
                    "e2 (halfW-trace)", "e3 (iso matched)", "psi"], body)]
    return "\n".join(out)
