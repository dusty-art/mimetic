"""Neural surrogate of the muscle simulator: activations -> skin vertices in microseconds, differentiably.

x(a) = LBS_jaw(rest, a_jaw) + B a + MLP(a)

The jaw is rigid and analytic, so it is computed exactly; the network only
learns the soft-tissue response (linear muscle basis plus a nonlinear
correction for saturation, contact and muscle-muscle interaction).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mimetic.face import anatomy as A
from mimetic.face.sim import jaw_rt


class MuscleSurrogate(nn.Module):
    def __init__(self, rest: torch.Tensor, w_jaw: torch.Tensor, n_act: int = A.N_ACT, hidden: int = 768,
                 tmj=(0.0, 0.3, -1.6)):
        super().__init__()
        self.N = rest.shape[0]
        self.n_act = n_act
        self.tmj = tuple(tmj)
        self.register_buffer("rest", rest.float())
        self.register_buffer("w_jaw", w_jaw.float())
        self.lin = nn.Linear(n_act, self.N * 3, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(n_act, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, self.N * 3))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def skinned(self, a_jaw):
        R, t = jaw_rt(a_jaw, self.tmj)
        Xj = torch.einsum("bij,nj->bni", R, self.rest) + t[:, None]
        w = self.w_jaw[None, :, None]
        return (1 - w) * self.rest[None] + w * Xj

    def soft(self, a):
        return (self.lin(a) + self.mlp(a)).view(a.shape[0], self.N, 3)

    def forward(self, a):
        """a (B, N_ACT) -> vertices (B, N, 3)."""
        shape = a.shape
        a = a.reshape(-1, shape[-1])
        x = self.skinned(a[:, -5:]) + self.soft(a)
        return x.view(*shape[:-1], self.N, 3)

    @classmethod
    def from_sim(cls, sim, **kw):
        return cls(sim.X.detach().cpu(), sim.w_jaw.detach().cpu(), tmj=sim.TMJ, **kw)


def sample_activations(B: int, generator: torch.Generator | None = None, real: torch.Tensor | None = None):
    """Mixture of activation patterns that covers what faces actually do.

    sparse independent muscles, bilaterally symmetric patterns, dense low-level
    tone, and (when given) perturbed activations fitted from real video.
    """
    g = generator
    M, J = A.N_SKIN_MUSCLES, len(A.JAW)
    out = torch.zeros(B, M + J)
    kind = torch.randint(0, 4 if real is not None else 3, (B,), generator=g)
    u = torch.rand(B, M + J, generator=g)
    active = torch.rand(B, M + J, generator=g) < 0.14
    sparse = (u ** 1.4) * active
    # symmetric: copy right-side values onto the left with noise
    sym = sparse.clone()
    names = A.MUSCLE_NAMES
    for i, n in enumerate(names):
        if n.endswith("_L") and not n.startswith("jaw"):
            j = names.index(n[:-2] + "_R")
            noise = (torch.rand(B, generator=g) - 0.5) * 0.3
            sym[:, i] = (sym[:, j] + noise * sym[:, j].clamp_min(0.1)).clamp(0, 1)
    dense = (torch.rand(B, M + J, generator=g) * 0.35) ** 2
    out = torch.where((kind == 0)[:, None], sparse, out)
    out = torch.where((kind == 1)[:, None], sym, out)
    out = torch.where((kind == 2)[:, None], dense, out)
    if real is not None:
        idx = torch.randint(0, len(real), (B,), generator=g)
        r = real[idx] + (torch.rand(B, M + J, generator=g) - 0.5) * 0.15
        out = torch.where((kind == 3)[:, None], r.clamp(0, 1), out)
    # Speech moves the jaw constantly; give it a broad distribution.
    jo = names.index("jaw_open")
    has_jaw = torch.rand(B, generator=g) < 0.75
    out[:, jo] = torch.where(has_jaw, torch.rand(B, generator=g) ** 1.3 * 0.8, out[:, jo])
    # jaw_close and lateral moves are rare and small
    for n, p, s in (("jaw_close", 0.15, 1.0), ("jaw_protrude", 0.1, 0.6), ("jaw_lateral_R", 0.05, 0.5),
                    ("jaw_lateral_L", 0.05, 0.5)):
        k = names.index(n)
        on = torch.rand(B, generator=g) < p
        out[:, k] = torch.where(on, torch.rand(B, generator=g) * s, out[:, k] * 0)
    return out.clamp(0, 1)
