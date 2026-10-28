"""Geometric articulation measures on simulated faces and the speech constraints built on them.

These are the speech-critical gestures viewers use to judge lip-sync: lips must
seal for p/b/m, the lower lip must meet the upper teeth for f/v, lips must
round for oo/w/sh, and the jaw must open for open vowels. The constraints are
stated from phonetics, not learned from data, so they hold even when the
tracked data under-represents them and even while an emotion pulls the
mouth elsewhere.
"""
from __future__ import annotations

import torch

from mimetic.data import phonemes as P

INNER_PAIRS = [(13, 14), (82, 87), (312, 317), (81, 178), (311, 402)]
CORNERS = (61, 291)
FRONT = [0, 17, 13, 14, 37, 267, 84, 314]


def measures(x: torch.Tensor) -> dict:
    """x (..., N, 3) -> dict of (...,) tensors in cm."""
    up = torch.stack([x[..., u, 1] for u, _ in INNER_PAIRS], -1)
    lo = torch.stack([x[..., l, 1] for _, l in INNER_PAIRS], -1)
    aperture = (up - lo).mean(-1)
    width = (x[..., CORNERS[1], :] - x[..., CORNERS[0], :]).norm(dim=-1)
    protrusion = x[..., FRONT, 2].mean(-1)
    tuck = x[..., 14, 2] - x[..., 13, 2]          # < 0 when the lower lip is behind the upper lip
    return {"aperture": aperture, "width": width, "protrusion": protrusion, "tuck": tuck}


class ArticulationLoss:
    def __init__(self, rest_x: torch.Tensor, jaw_index: int):
        r = measures(rest_x)
        self.W0 = r["width"].item()
        self.P0 = r["protrusion"].item()
        self.jaw = jaw_index
        self.artic = torch.tensor(P.ARTIC)       # (N_PHONES, 7)

    def __call__(self, x, acts, phone, ppos, mask, arousal=None):
        """x (B,T,N,3), acts (B,T,A), phone (B,T) ids, ppos (B,T) position in phone, mask (B,T)."""
        art = self.artic.to(x.device)[phone]                    # (B,T,7)
        close, labdent, rnd, spread, opn, dental, crit = art.unbind(-1)
        centre = (1 - (2 * ppos - 1).abs()).clamp(0, 1)
        core = (centre > 0.35).float() * mask
        m = measures(x)
        ap, wd, pr, tk = m["aperture"], m["width"], m["protrusion"], m["tuck"]
        l_close = close * torch.relu(ap - 0.04) ** 2
        l_lab = labdent * (torch.relu(ap - 0.15) ** 2 + torch.relu(tk + 0.08) ** 2)
        l_round = rnd * (torch.relu(wd - self.W0 * (1 - 0.10 * rnd)) ** 2
                         + torch.relu(self.P0 + 0.10 * rnd - pr) ** 2)
        jaw = acts[..., self.jaw]
        l_open = opn * torch.relu(0.35 * opn - jaw) ** 2
        per = 25.0 * l_close + 10.0 * l_lab + 10.0 * l_round + 2.0 * l_open
        loss = (per * (0.5 + crit) * core).sum() / core.sum().clamp_min(1)
        # Diagnostics
        with torch.no_grad():
            bil = (close > 0.5) & (core > 0)
            stats = {"closure_rate": ((ap < 0.08) & bil).sum().item() / max(1, bil.sum().item())}
        return loss, stats


def bilabial_closure_rate(x, phone, ppos, threshold=0.08):
    """Fraction of bilabial phoneme occurrences whose minimum lip aperture falls below `threshold` cm."""
    m = measures(x)["aperture"]
    ids = torch.tensor([P.PID[p] for p in ("P", "B", "M")], device=phone.device)
    bil = torch.isin(phone, ids)
    hits = tot = 0
    # Walk contiguous runs
    for b in range(phone.shape[0]):
        t = 0
        T = phone.shape[1]
        while t < T:
            if bil[b, t]:
                s = t
                while t < T and bil[b, t]:
                    t += 1
                tot += 1
                hits += int(m[b, s:t].min().item() < threshold)
            else:
                t += 1
    return hits / max(1, tot), tot
