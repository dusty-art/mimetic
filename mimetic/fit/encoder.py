"""Physics-decoder autoencoder: tracked face motion -> muscle activations.

The decoder is the (frozen) neural surrogate of the muscle simulator, so the
only way for the encoder to reproduce a tracked face is to find muscle
activations that physically produce it. Effort (L1) and temporal smoothness
priors pick the most economical activation among the many that fit, in line
with minimum-effort accounts of motor control.

An optional calibration layer adds a per-muscle corrective displacement field,
masked to that muscle's anatomical territory, learned from all speakers: the
simulator's anatomy is the prior, real faces refine it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimetic.face import anatomy as A
from mimetic.face import canonical as C


def fit_weights(X: np.ndarray) -> torch.Tensor:
    """Per-vertex (N,3) weights for the fitting loss."""
    n = len(X)
    w = np.full(n, 1.0)
    lips = C.UPPER_INNER + C.LOWER_INNER + C.UPPER_OUTER + C.LOWER_OUTER
    lids = C.R_UPPER_LID + C.R_LOWER_LID + C.L_UPPER_LID + C.L_LOWER_LID
    w[C.RIGID_ANCHORS] = 0.2
    x, y = X[:, 0], X[:, 1]
    w[(np.abs(x) > 2.3) & (np.abs(x) < 6.0) & (y > -2.5) & (y < 1.5)] = 2.5     # cheeks (smile raises them)
    w[(y > 3.4) & (y < 6.5)] = np.maximum(w[(y > 3.4) & (y < 6.5)], 2.0)       # brow band
    w[C.R_BROW + C.L_BROW] = 4.0
    w[lids] = 6.0
    w[lips] = 5.0
    W = np.stack([w, w, 0.35 * w], -1)      # monocular depth is the least reliable axis
    return torch.tensor(W / W.mean(), dtype=torch.float32)


class TemporalEncoder(nn.Module):
    """PCA displacement coefficients (B,T,K) -> muscle activations (B,T,N_ACT)."""

    def __init__(self, k_in: int = 96, hidden: int = 384, n_act: int = A.N_ACT):
        super().__init__()
        self.inp = nn.Linear(k_in, hidden)
        self.blocks = nn.ModuleList()
        for d in (1, 2, 4, 1):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d), nn.GELU(),
                nn.Conv1d(hidden, hidden, 1)))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.blocks])
        self.out = nn.Linear(hidden, n_act)
        nn.init.constant_(self.out.bias, -3.0)    # start near rest: low activations

    def forward(self, c):
        h = self.inp(c)
        for blk, ln in zip(self.blocks, self.norms):
            h = h + blk(ln(h).transpose(1, 2)).transpose(1, 2)
        return torch.sigmoid(self.out(h))


class Calibration(nn.Module):
    """Per-muscle corrective displacement fields restricted to each muscle's territory."""

    def __init__(self, sim_X: np.ndarray, muscle_weights: np.ndarray, w_jaw: np.ndarray, face):
        super().__init__()
        N = len(sim_X)
        masks = []
        nb = face.neighbors
        for m in range(muscle_weights.shape[0]):
            m0 = muscle_weights[m] > 0.03
            for _ in range(2):    # dilate two rings
                m1 = m0.copy()
                for v in np.nonzero(m0)[0]:
                    m1[nb[v]] = True
                m0 = m1
            masks.append(m0)
        lower = w_jaw > 0.05
        for _ in A.JAW:
            masks.append(lower)
        self.register_buffer("mask", torch.tensor(np.stack(masks), dtype=torch.float32)[..., None])  # (M,N,1)
        self.field = nn.Parameter(torch.zeros(len(masks), N, 3))

    def forward(self, a):
        return torch.einsum("...m,mnc->...nc", a, self.field * self.mask)

    def penalty(self):
        return (self.field * self.mask).pow(2).mean()
