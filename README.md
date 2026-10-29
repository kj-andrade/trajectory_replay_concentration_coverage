# Trajectory replay in sequential flow matching

Code for the preprint *Trajectory Replay in Sequential Flow Matching:
Concentration & Coverage* by Karan J. Andrade and Felipe Tobar (Imperial
College London). The paper is in
[`paper/`](paper/TrajReplayConcentrationCoverage_preprint.pdf).

When flow matching is trained on sequences, the best velocity field you can
fit to a finite set of stored transitions (with no limit on model capacity) is
a kernel-weighted average of those transitions. As the kernel sharpens, the
average collapses onto one stored transition and the model just replays its
training data. The paper separates two questions about this. The first is when
replay happens. We show it is governed by a single concentration parameter,
κ = d·V/σ², and sets in once κ passes a threshold of order log M, where M is
the number of stored transitions. The second is whether avoiding replay is
enough to predict well, and the answer is no. With a covariance that is
constant in flow time, the field always stays inside the convex hull of the
stored velocity targets. That hull puts a floor on the error, and no bandwidth
or covariance choice can get below it.

The experiments check both results on Lorenz-63 and Lorenz-96. A final
experiment trains small MLPs to see how much of this behaviour real networks
inherit.

## What's here

```
paper/                        the preprint (PDF)
fm_replay/                    closed-form experiments, paper Sections 6.1-6.3
  experiments/                onset.py, coverage.py, anisotropy.py
  figures/                    one plotting module per figure
scripts/                      command-line entry points: run, plot, verify
configs/final.yaml            every setting used for the paper
experiments/finite_capacity/  trained-network experiment, Section 6.4
```

| Paper figure | What it tests | Code |
| --- | --- | --- |
| Fig. 2 | Replay onset across dimension, window order, interpolant and bank size | `fm_replay/experiments/onset.py` |
| Fig. 3 | Coverage and the convex-hull error floor | `fm_replay/experiments/coverage.py` |
| Fig. 4 | Covariance shape at matched concentration | `fm_replay/experiments/anisotropy.py` |
| Figs 5-7 | Trained MLPs compared with the closed-form field | `experiments/finite_capacity/` |

The closed-form experiments train no networks. They evaluate the kernel weights
and the resulting velocity field exactly. The ground truth is the true one-step
displacement, obtained by integrating the ODE.

## Running it

You need Python 3.10 or later. The results in the paper were produced on CPU
with Python 3.11, NumPy 2.3, SciPy 1.16, Matplotlib 3.10 and PyTorch 2.11.

```bash
pip install -e .                # closed-form experiments
pip install -e ".[networks]"    # adds PyTorch for the network experiment
```

### Closed-form experiments (Figs 2-4)

Run these from the repository root:

```bash
python -m scripts.run --exp onset      --config configs/final.yaml
python -m scripts.run --exp coverage   --config configs/final.yaml
python -m scripts.run --exp anisotropy --config configs/final.yaml
python -m scripts.plot --all --results results --out figures
python -m scripts.verify
```

`run_full.sh` runs the first four steps in order. On a 16-thread desktop, the
three experiments took about 35, 90 and 40 minutes. If you add `--quick`, each
grid shrinks enough to finish in a few minutes, which is useful for checking
that everything runs. The numbers it produces are not the ones in the paper.

Each experiment writes `results/<name>/rows.csv` along with the resolved
config, a `pip freeze` and timings. The plotting script only reads `rows.csv`,
so you can redraw figures without rerunning anything. `scripts.verify`
re-checks the solver constraints and the Monte Carlo cross-checks against the
saved rows. Simulated trajectories are cached in `results/cache`, which grows
to a few GB. You can delete it at any time; the only cost is recomputing the
trajectories on the next run.

### Network experiment (Figs 5-7)

Run these from `experiments/finite_capacity/`:

```bash
python -m unittest discover -s tests -v   # closed form and setup checks
python smoke_test.py                      # one short 5,000-step run
python run_grid.py                        # 30 runs: 5 bandwidths x 2 widths x 3 seeds
python make_figures.py
```

On CPU, with six runs in parallel, the grid took about two hours. Before it
starts, `run_grid.py` times a few training steps. If the full grid would take
longer than `TIME_BUDGET_HOURS` in `config/grid.yaml`, it drops seeds first,
then training steps, then bandwidths. To run the full grid on a slower
machine, raise the budget.

## Reproducibility

Every random draw is keyed by system, seed and purpose through NumPy's
`SeedSequence`. Results therefore don't depend on the order things run in,
and a run can be split across processes. Memory banks of different sizes are
nested subsets of one permutation, so comparisons across bank size are
paired. The query sets are fixed across bandwidths and covariance shapes.

Reported values are the median over 500 queries within each seed, then the
median across five seeds (three for the network experiment), with the seed
min-max as the error bar. Ratios and differences are computed within each
seed before aggregating. Thresholds and tolerances live in the config files
and were fixed before the final runs.

## Citation

```bibtex
@misc{andrade2026replay,
  title  = {Trajectory Replay in Sequential Flow Matching: Concentration \& Coverage},
  author = {Andrade, Karan J. and Tobar, Felipe},
  year   = {2026},
  note   = {Preprint}
}
```

## License

The code is released under the [MIT License](LICENSE). The paper in `paper/`
is not covered by this license; all rights to it remain with the authors.
