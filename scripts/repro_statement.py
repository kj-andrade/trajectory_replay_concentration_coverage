"""Generate the reproducibility statement from a run's own artefacts.

Reads results/<exp>/{config,env,timing} and emits a reproducibility
statement in Markdown, so the statement cannot drift from the run that
actually produced the numbers.

    python -m scripts.repro_statement --results results --out figures
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import yaml

from fm_replay.rng import BASE_SEED, STREAMS
from fm_replay.data import BURN, GROUND_TRUTH_SUBSTEPS, VSCALE_SAMPLES
from fm_replay.coverage import LAMBDA, MAXITER_MULT, SIMPLEX_TOL
from fm_replay.systems import all_systems, DETERMINISTIC_IC

EXPERIMENTS = ("onset", "coverage", "anisotropy")


def _read(path, loader):
    try:
        return loader(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _env_line(env, key):
    if not env:
        return "unavailable"
    for line in env.splitlines():
        if line.startswith(key):
            return line.split(None, 1)[1].strip()
    return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)

    env = None
    for name in EXPERIMENTS:
        env = env or _read(res / name / "env.txt", lambda t: t)

    L = ["# Reproducibility statement", "",
         "## Seeds", "",
         f"All randomness derives from a single base entropy "
         f"`BASE_SEED = {BASE_SEED}` through `numpy.random.SeedSequence`. "
         f"No call uses global RNG state. A generator is addressed by "
         f"`(system_id, seed, stream)`, optionally extended by a fixed "
         f"sub-key, so results do not depend on the order in which cells "
         f"execute or on how the work is parallelised. Seeds are "
         f"`{{0, 1, 2, 3, 4}}` (n = 5).", "",
         "| stream | id | used for |", "|---|---|---|"]
    use = {"traj": "training initial condition and burn-in perturbation",
           "test_traj": "independent test-trajectory initial condition",
           "bank": "memory-bank subsampling",
           "query_val": "validation query set",
           "query_test": "test query set",
           "noise": "Gaussian noise eps for training-law queries"}
    for k, v in STREAMS.items():
        L.append(f"| `{k}` | {v} | {use.get(k, '')} |")

    L += ["", "Banks of different `M` are nested subsets of one permutation "
          "per `(system, seed)`, so comparisons across `M` are paired. Query "
          "families are drawn once per `(system, seed, family)` and reused "
          "across every `M`, `kappa` and geometry.", "",
          "## Systems", "",
          "| id | name | d | dt | stride | dtau | initial condition |",
          "|---|---|---|---|---|---|---|"]
    for s in sorted(all_systems().values(), key=lambda z: z.id):
        if s.kind == "gauss":
            L.append(f"| {s.id} | {s.name} | {s.d} | -- | -- | -- | "
                     f"iid N(0, I) bank, no dynamics |")
        else:
            ic = getattr(s.ic, "__name__", "--")
            note = " (deterministic; per-seed variation from the burn-in " \
                   "perturbation)" if s.id in DETERMINISTIC_IC else ""
            L.append(f"| {s.id} | {s.name} | {s.d} | {s.dt} | {s.stride} | "
                     f"{s.dtau:g} | `{ic}`{note} |")

    L += ["", f"Integrator RK4 throughout. Burn-in {BURN:,} steps. Ground "
          f"truth is the exact one-step displacement of the true flow, "
          f"obtained with RK4 over `dtau` in {GROUND_TRUTH_SUBSTEPS} "
          f"substeps, defined everywhere including off the attractor. "
          f"States are standardised per coordinate by the training mean and "
          f"standard deviation; errors are normalised by `v_scale`, the "
          f"median speed of the true field over {VSCALE_SAMPLES:,} "
          f"test-trajectory states.", "",
          "## Per-experiment settings", ""]

    for name in EXPERIMENTS:
        cfg = _read(res / name / "config.yaml", yaml.safe_load)
        if not cfg:
            L += [f"### {name}", "", "_not run_", ""]
            continue
        r = cfg.get("resolved", cfg)
        L += [f"### {name}" + ("  **(quick smoke run, not publication "
                               "numbers)**" if cfg.get("quick") else ""), "",
              "```yaml", yaml.safe_dump(r, sort_keys=True,
                                        default_flow_style=False).rstrip(),
              "```", ""]

    L += ["## Aggregation", "",
          "Per seed, the median over test queries. Across seeds, the median "
          "of those per-seed medians; error bars and bracketed ranges are "
          "min-max over the five seeds. Paired quantities (ratios, "
          "differences) are formed per seed first and only then aggregated, "
          "so a ratio is never a ratio of medians taken over different "
          "banks.", "",
          "## Solver settings", "",
          f"The convex-hull floor is solved as a non-negative least squares "
          f"problem with the simplex constraint imposed by a weighted row, "
          f"`lambda = {LAMBDA:g}`, `maxiter = {MAXITER_MULT}M`. The realised "
          f"`|sum alpha - 1|` is recorded for every query and asserted below "
          f"{SIMPLEX_TOL:g}.", "",
          "## Declared thresholds", ""]

    cfg = _read(res / "anisotropy" / "config.yaml", yaml.safe_load)
    tol = (cfg or {}).get("resolved", {}).get("tol", "--")
    L += [f"The matched-concentration tolerance `tol = {tol}` is declared in "
          f"`configs/final.yaml` before the run and snapshotted next to the "
          f"results. A cell counts as matched only when the whole seed range "
          f"lies inside the band.", "",
          "## Environment", "", "```"]
    for k in ("python", "numpy", "scipy", "platform", "processor",
              "cpu_count", "OMP_NUM_THREADS", "git_commit"):
        L.append(f"{k:<16}{_env_line(env, k)}")
    L += ["```", "", "Wall-clock time:", "", "| experiment | seconds |",
          "|---|---|"]
    total = 0.0
    for name in EXPERIMENTS:
        t = _read(res / name / "timing.json", json.loads)
        if t:
            total += t["total_seconds"]
            L.append(f"| {name} | {t['total_seconds']:.0f} |")
    L.append(f"| **total** | **{total:.0f}** |")

    L += ["", "## Code", "",
          "```", "python -m scripts.run  --exp onset      --config configs/final.yaml",
          "python -m scripts.run  --exp coverage   --config configs/final.yaml",
          "python -m scripts.run  --exp anisotropy --config configs/final.yaml",
          "python -m scripts.plot --all --results results --out figures",
          "python -m scripts.verify --results results", "```", ""]

    out.mkdir(parents=True, exist_ok=True)
    p = out / "reproducibility_statement.md"
    p.write_text("\n".join(L), encoding="utf-8")
    print(f"  {p}")


if __name__ == "__main__":
    main()
