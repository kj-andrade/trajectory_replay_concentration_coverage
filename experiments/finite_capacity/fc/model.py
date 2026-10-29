"""The trained velocity network.

Input is [x, emb(t)] with emb(t) = [sin(2 pi k t), cos(2 pi k t)] for
k = 1..16, three hidden layers of width W with SiLU activations, and a linear
output of dimension d.  Framework-default initialisation, no normalisation
layers, no residual connections: the point of the experiment is what an
ordinary regression network does with this objective, not what a tuned one
can be made to do.
"""
from __future__ import annotations

import math

import torch
from torch import nn

N_FREQ = 16


def time_embedding(t, n_freq=N_FREQ):
    """(B,) flow times -> (B, 2 n_freq) Fourier features."""
    k = torch.arange(1, n_freq + 1, device=t.device, dtype=t.dtype)
    ang = 2.0 * math.pi * t[:, None] * k[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class VelocityMLP(nn.Module):
    def __init__(self, d, width, n_freq=N_FREQ, depth=3):
        super().__init__()
        self.d = d
        self.n_freq = n_freq
        dims = [d + 2 * n_freq] + [width] * depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        layers += [nn.Linear(dims[-1], d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, time_embedding(t, self.n_freq)], dim=1))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def make_model(d, width, seed, device="cpu"):
    """Framework-default initialisation under a dedicated generator.

    The initialisation is drawn from the run seed but from a generator of its
    own, so that changing the number of training steps cannot change the
    initial weights.
    """
    g = torch.Generator(device="cpu").manual_seed(1_000_000 + int(seed))
    prev = torch.random.get_rng_state()
    torch.random.set_rng_state(g.get_state())
    try:
        m = VelocityMLP(d, width)
    finally:
        torch.random.set_rng_state(prev)
    return m.to(device)
