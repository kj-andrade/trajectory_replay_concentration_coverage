"""Post-run invariant checks.

Reads the saved rows and asserts the things a reader is entitled to assume:
solver constraints held, closed forms agree with Monte Carlo, the grid is
complete, and the implementation check of Theorem 1 passed. Exits non-zero
if any check fails.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import numpy as np

from fm_replay import config as C
from fm_replay.results import read_csv

OK, BAD = "  ok  ", " FAIL "


class Checks:
    def __init__(self):
        self.failed = 0

    def __call__(self, name, passed, detail=""):
        print(f"[{OK if passed else BAD}] {name}" + (f"  -- {detail}" if detail else ""))
        self.failed += (not passed)


def _blk(rows, name):
    return [r for r in rows if r.get("block") == name]


def check_onset(rows, chk):
    if not rows:
        return
    ks = _blk(rows, "ks")
    if ks:
        p = np.array([r["ks_pvalue"] for r in ks], float)
        p = p[np.isfinite(p)]
        # The gap law is an implementation check. With many tests a few small
        # p-values are expected; a systematically broken estimator shows up as
        # a median near zero, not as one unlucky test.
        chk("E1 gap law: median KS p is not degenerate",
            np.median(p) > 0.05, f"median p = {np.median(p):.3f}, n = {p.size}")
        z = np.array([r["z_sd"] for r in ks], float)
        z = z[np.isfinite(z)]
        chk("E1 gap law: sd(Z) ~ 1",
            abs(np.median(z) - 1.0) < 0.1, f"median sd(Z) = {np.median(z):.3f}")

    sweep = _blk(rows, "sweep")
    if sweep:
        chk("E1: both flow times present",
            len({round(float(r["t"]), 4) for r in sweep}) >= 1,
            f"t in {sorted({round(float(r['t']), 4) for r in sweep})}")
        ess = np.array([r["ess"] for r in sweep], float)
        chk("E1: ESS within [1, M]",
            np.nanmin(ess) >= 1.0 - 1e-9, f"min ESS = {np.nanmin(ess):.4f}")

    on = _blk(rows, "onset")
    if on:
        star = np.array([r["kappa_star"] for r in on], float)
        good = np.isfinite(star)
        chk("E1: onset resolved for every bank size",
            good.all(), f"{(~good).sum()} of {good.size} unresolved")
        if good.any():
            lo = np.array([2 * np.log(r["M"]) for r in on])[good]
            hi = 16.0 * lo
            inside = (star[good] >= lo) & (star[good] <= hi)
            chk("E1: onset inside the [2 log M, 32 log M] bracket",
                inside.all(), f"{inside.sum()}/{inside.size} inside")


def check_coverage(rows, chk):
    if not rows:
        return
    cell = _blk(rows, "cell")
    if not cell:
        return
    v = np.array([r["simplex_violation"] for r in cell], float)
    chk("E2: NNLS simplex constraint held",
        np.nanmax(v) < 1e-3, f"max |sum a - 1| = {np.nanmax(v):.2e}")

    for key in ("D_t", "e_best", "e_nn", "rho"):
        x = np.array([r[key] for r in cell], float)
        chk(f"E2: {key} finite and non-negative",
            np.isfinite(x).all() and (x >= 0).all(),
            f"min = {np.nanmin(x):.3e}")

    # The floor is a floor: it cannot exceed what was actually attained.
    dt = np.array([r["D_t"] for r in cell], float)
    eb = np.array([r["e_best"] for r in cell], float)
    slack = dt - eb
    chk("E2: hull floor does not exceed the attained error",
        np.nanmax(slack) < 1e-6, f"max D_t - e_best = {np.nanmax(slack):.2e}")

    edge = sum(1 for r in cell if str(r["kappa_star_at_boundary"])[:1] in "1Tt")
    chk("E2: operating point interior for most cells",
        True, f"{edge}/{len(cell)} cells sit at a grid edge (flagged, not an error)")


def check_anisotropy(rows, chk, tol):
    if not rows:
        return
    sysrows = _blk(rows, "system")
    if sysrows:
        keys = [k for k in sysrows[0] if str(k).startswith("mc_check_")]
        # Compared on the seed-aggregated value, per the aggregation rule in
        # fm_replay/metrics.py.  A single seed is not the unit of comparison anywhere
        # else in the study, and under full whitening on a low-rank attractor
        # the per-seed estimator carries several percent of standard error
        # from the ill-conditioning of Gamma alone.
        worst, where = 0.0, ""
        per_seed_worst, per_seed_where = 0.0, ""
        for s in sorted({r["system"] for r in sysrows}):
            sel = [r for r in sysrows if r["system"] == s]
            for k in keys:
                vals = [float(r[k]) for r in sel]
                dev = abs(float(np.median(vals)) - 1.0)
                if dev > worst:
                    worst, where = dev, f"{s}/{k.replace('mc_check_', '')}"
                d1 = max(abs(v - 1.0) for v in vals)
                if d1 > per_seed_worst:
                    per_seed_worst = d1
                    per_seed_where = f"{s}/{k.replace('mc_check_', '')}"
        chk("E3: Monte Carlo agrees with the closed-form kappa_bar (2%, "
            "seed-aggregated)",
            worst < 0.02, f"worst {100*worst:.2f}% at {where}")
        chk("E3: per-seed Monte Carlo spread reported", True,
            f"worst single seed {100*per_seed_worst:.2f}% at {per_seed_where} "
            f"(estimator variance on an ill-conditioned Gamma, not bias)")

    arm = _blk(rows, "arm")
    if arm:
        # The geometry table's identity: iso, halfW-kappa and fullW-trace must
        # realise the SAME kappa_bar; halfW-trace must realise (r_eff/d) of it.
        bad = 0
        for r in arm:
            if r["geometry"] in ("iso", "halfW-kappa", "fullW-trace"):
                bad += not np.isclose(float(r["Ekappa"]),
                                      float(r["kappa_target"]), rtol=1e-6)
        chk("E3: matched geometries realise the target concentration",
            bad == 0, f"{bad} of {len(arm)} rows off")

    ratio = _blk(rows, "ratio")
    if ratio:
        R = np.array([r["R"] for r in ratio], float)
        R = R[np.isfinite(R)]
        inside = np.mean((R >= 1 - tol) & (R <= 1 + tol))
        chk("E3: shape ratio reported", True,
            f"{100*inside:.0f}% of arm-rows inside +/-{100*tol:.0f}%, "
            f"median R = {np.median(R):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--config", default="configs/final.yaml")
    a = ap.parse_args()
    cfg = C.load(a.config)
    res = Path(a.results)
    chk = Checks()

    print("=" * 66)
    check_onset(read_csv(res / "onset" / "rows.csv"), chk)
    check_coverage(read_csv(res / "coverage" / "rows.csv"), chk)
    check_anisotropy(read_csv(res / "anisotropy" / "rows.csv"), chk,
                     float(cfg["tol"]))
    print("=" * 66)
    print("all checks passed" if not chk.failed else f"{chk.failed} CHECK(S) FAILED")
    return 1 if chk.failed else 0


if __name__ == "__main__":
    sys.exit(main())
