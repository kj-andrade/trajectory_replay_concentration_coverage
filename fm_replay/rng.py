"""Seed discipline.

All randomness in the package derives from one base entropy through
``numpy.random.SeedSequence``.  Nothing touches global RNG state, and every
generator is addressed by (system_id, seed, stream), so results do not depend
on the order in which cells are executed or on how work is parallelised.
"""
from __future__ import annotations

import numpy as np

BASE_SEED = 20260922

# Stream ids.  Fixed integers: changing one changes only that draw.
TRAJ = 0          # training initial condition and burn-in perturbation
TEST_TRAJ = 1     # independent test-trajectory initial condition
BANK = 2          # memory-bank subsampling
QUERY_VAL = 3     # validation query set
QUERY_TEST = 4    # test query set
NOISE = 5         # Gaussian noise eps for training-law queries

STREAMS = dict(traj=TRAJ, test_traj=TEST_TRAJ, bank=BANK,
               query_val=QUERY_VAL, query_test=QUERY_TEST, noise=NOISE)


def rng(system_id, seed, stream, sub=(), base=BASE_SEED):
    """Generator for one (system, seed, stream) cell.

    ``stream`` may be the integer id or its name from STREAMS.  ``sub``
    extends the spawn key with further fixed integers, which is how
    independent sub-draws (one per query family, say) are addressed: adding
    or removing one sub-draw never shifts any other.  Sub-keys must be
    stable integers, never hashes of strings, whose values vary per process.
    """
    if isinstance(stream, str):
        stream = STREAMS[stream]
    key = (int(system_id), int(seed), int(stream)) + tuple(int(s) for s in sub)
    return np.random.default_rng(
        np.random.SeedSequence(entropy=base, spawn_key=key))
