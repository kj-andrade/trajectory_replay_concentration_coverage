"""Correctness checks for the closed-form field and the network setup.

Run from experiments/finite_capacity with

    python -m unittest discover -s tests -v

Each test states what a failure means.  A failure here is a bug in this
module, not evidence about the theory.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fc import load_config
from fc.closed_form import closed_form, conditional_loss
from fc.model import make_model
from fc.problem import build, _key

from fm_replay.kernel import sq_distances, estimate

CFG = load_config()
P, SCAL = build(CFG, verbose=False)
M, D = int(SCAL["M"]), int(SCAL["d"])
MU0, MU1, U = P["C"], P["C"] + P["U"], P["U"]


class ClosedFormTests(unittest.TestCase):

    def test_matches_repository_closed_form(self):
        """fc.closed_form against fm_replay.kernel, under the same sigma."""
        r = np.random.default_rng(11)
        x = MU0[r.integers(0, M, 100)] + 0.5 * r.standard_normal((100, D))
        for t in (0.25, 0.5, 0.75):
            for lam in CFG["lambdas"]:
                s2 = SCAL["sigma2"][str(float(lam))]
                mu_t = (1 - t) * MU0 + t * MU1
                v_a, alpha, ess_a = closed_form(x, t, MU0, MU1, U, s2)
                v_b, ess_b, _ = estimate(sq_distances(x, mu_t), U, s2)
                rel = (np.linalg.norm(v_a - v_b, axis=1)
                       / np.maximum(np.linalg.norm(v_b, axis=1), 1e-12))
                self.assertLess(float(rel.max()), 1e-6,
                                f"v* disagrees at t={t}, lambda={lam}")
                self.assertLess(float(np.abs(ess_a - ess_b).max()), 1e-6)

    def test_weights_are_a_probability_vector(self):
        """Rows sum to one and stay finite at the most concentrated cell."""
        lam = max(CFG["lambdas"])
        s2 = SCAL["sigma2"][str(float(lam))]
        for t in CFG["t_eval"]:
            x = P[_key("IN", "X", CFG["lambdas"].index(lam),
                       list(CFG["t_eval"]).index(t))]
            _, alpha, ess = closed_form(x, float(t), MU0, MU1, U, s2)
            self.assertTrue(np.isfinite(alpha).all(), "non-finite alpha")
            self.assertTrue(np.isfinite(ess).all(), "non-finite ESS")
            self.assertTrue((alpha >= 0).all(), "negative alpha")
            self.assertLess(float(np.abs(alpha.sum(1) - 1.0).max()), 1e-12)
            self.assertGreaterEqual(float(ess.min()), 1.0 - 1e-9)

    def test_v_star_beats_zero_and_initialisation(self):
        """v* is the minimiser, so nothing untrained can beat it."""
        for li, lam in enumerate(CFG["lambdas"]):
            s2 = SCAL["sigma2"][str(float(lam))]
            j = P[_key("dL", "j", li)].astype(np.intp)[:8192]
            L_star = float(P[_key("dL", "per_star", li)][:8192].mean())

            t = P[_key("dL", "t", li)][:8192]
            r = np.random.default_rng(4000 + li)
            x = MU0[j] + t[:, None] * (MU1 - MU0)[j] + np.sqrt(s2) * \
                r.standard_normal((len(j), D))

            L_zero, _ = conditional_loss(np.zeros_like(U[j]), U, j)
            self.assertLessEqual(L_star, L_zero + 1e-12,
                                 f"v* worse than the zero field at lambda={lam}")

            net = make_model(D, 512, seed=0)
            with torch.no_grad():
                v0 = net(torch.as_tensor(x, dtype=torch.float32),
                         torch.as_tensor(t, dtype=torch.float32)).double().numpy()
            L_init, _ = conditional_loss(v0, U, j)
            self.assertLessEqual(L_star, L_init + 1e-12,
                                 f"v* worse than initialisation at lambda={lam}")

    def test_tiny_bandwidth_recovers_the_generating_window(self):
        """At lambda = 1000 the generating window must dominate."""
        lam = 1000.0
        s2 = D * SCAL["V_half"] / (lam * SCAL["log_M"])
        i_gen = P["i_gen"].astype(np.intp)
        r = np.random.default_rng(77)
        for t in (0.25, 0.5, 0.75):
            eps = r.standard_normal((len(i_gen), D))
            x = (1 - t) * MU0[i_gen] + t * MU1[i_gen] + np.sqrt(s2) * eps
            _, alpha, _ = closed_form(x, t, MU0, MU1, U, s2)
            hit = float((alpha.argmax(1) == i_gen).mean())
            self.assertGreater(hit, 0.99, f"i_star = i_gen only {hit:.3f} at t={t}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
