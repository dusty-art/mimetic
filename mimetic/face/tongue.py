"""Articulatory tongue: phoneme-driven tip, blade and dorsum, in the mandible's frame.

No talking-face corpus records the tongue (trackers cannot see it), so its
targets come from phonetics instead of data:
  alveolars  t d n l s z  tip up to the ridge behind the upper teeth
  dentals    th dh        tip forward between the teeth (visible)
  postalv.   sh zh ch jh  blade up, tip slightly retracted
  velars     k g ng       dorsum (back) up against the soft palate
  r          tip raised and curled back
  front vowels iy ih ey y  front of the body high
  back vowels  uw uh ow    back of the body high
  open vowels  aa ao ae aw low
Targets blend with anticipatory smoothing (the tongue starts moving toward a
consonant before it sounds), as coarticulation does in real speech.

Parameters (cm, mandible frame): tip_up, tip_fwd, front_up, back_up, low.
"""
from __future__ import annotations

import numpy as np

from mimetic.data import phonemes as P

PARAMS = ["tip_up", "tip_fwd", "front_up", "back_up", "low"]

_T = {
    "SIL": (0.0, 0.0, 0.0, 0.0, 0.0),
    "T": (0.70, -0.05, 0.15, 0.0, 0.0), "D": (0.70, -0.05, 0.15, 0.0, 0.0), "N": (0.70, -0.05, 0.15, 0.0, 0.0),
    "L": (0.75, 0.0, 0.0, -0.05, 0.05), "S": (0.55, -0.05, 0.25, 0.0, 0.0), "Z": (0.55, -0.05, 0.25, 0.0, 0.0),
    "TH": (0.22, 0.80, 0.1, 0.0, 0.0), "DH": (0.22, 0.70, 0.1, 0.0, 0.0),
    "SH": (0.55, -0.20, 0.40, 0.05, 0.0), "ZH": (0.55, -0.20, 0.40, 0.05, 0.0),
    "CH": (0.65, -0.15, 0.35, 0.05, 0.0), "JH": (0.65, -0.15, 0.35, 0.05, 0.0),
    "K": (-0.05, -0.10, 0.0, 0.60, 0.0), "G": (-0.05, -0.10, 0.0, 0.60, 0.0), "NG": (-0.05, -0.10, 0.0, 0.55, 0.0),
    "R": (0.45, -0.30, 0.10, 0.20, 0.0), "Y": (0.0, 0.0, 0.55, 0.10, 0.0),
    "IY": (-0.05, 0.05, 0.50, 0.15, 0.0), "IH": (-0.05, 0.0, 0.40, 0.10, 0.0), "EY": (-0.05, 0.0, 0.35, 0.05, 0.05),
    "EH": (-0.05, 0.0, 0.20, 0.0, 0.10), "AE": (-0.05, 0.05, 0.10, -0.05, 0.25),
    "AA": (-0.05, -0.10, -0.05, -0.05, 0.30), "AO": (-0.05, -0.15, -0.05, 0.05, 0.25), "AW": (-0.05, -0.10, 0.0, 0.05, 0.25),
    "AY": (-0.05, -0.05, 0.10, 0.0, 0.20), "AH": (-0.05, -0.05, 0.05, 0.05, 0.15), "ER": (0.30, -0.25, 0.10, 0.20, 0.05),
    "UW": (-0.05, -0.15, 0.05, 0.45, 0.0), "UH": (-0.05, -0.10, 0.05, 0.35, 0.05), "OW": (-0.05, -0.10, 0.0, 0.35, 0.10),
    "OY": (-0.05, -0.10, 0.10, 0.30, 0.10),
    "B": (0.0, 0.0, 0.0, 0.0, 0.0), "P": (0.0, 0.0, 0.0, 0.0, 0.0), "M": (0.0, 0.0, 0.0, 0.0, 0.0),
    "F": (0.0, 0.0, 0.05, 0.0, 0.0), "V": (0.0, 0.0, 0.05, 0.0, 0.0), "W": (-0.05, -0.10, 0.0, 0.40, 0.0),
    "HH": (0.0, 0.0, 0.0, 0.0, 0.10),
}
# Critical articulations must be reached; vowels and labials leave the tongue free to anticipate.
_CRIT = {p: 1.0 for p in ("T", "D", "N", "L", "S", "Z", "TH", "DH", "SH", "ZH", "CH", "JH", "K", "G", "NG", "R")}
TARGETS = np.array([_T[p] for p in P.PHONES], np.float32)          # (N_PHONES, 5)
CRIT = np.array([_CRIT.get(p, 0.35) for p in P.PHONES], np.float32)


def track_from_targets(target: np.ndarray, crit: np.ndarray, fps: float, lead_s: float = 0.04,
                       tau_s: float = 0.045) -> np.ndarray:
    """Coarticulated tongue trajectory: critical gestures dominate, others blend; anticipatory, zero-phase."""
    T = len(target)
    lead = int(round(lead_s * fps))
    tgt = np.concatenate([target[lead:], np.repeat(target[-1:], lead, 0)]) if lead and T > lead else target
    w = np.concatenate([crit[lead:], np.repeat(crit[-1:], lead, 0)]) if lead and T > lead else crit
    a = 1 - np.exp(-1 / (tau_s * fps))
    out = np.zeros_like(tgt)
    for direction in (1, -1):                                      # forward + backward pass = zero phase
        x = tgt[0].copy() if direction == 1 else out[-1].copy()
        rng = range(T) if direction == 1 else range(T - 1, -1, -1)
        buf = np.zeros_like(tgt)
        for t in rng:
            src = tgt[t] if direction == 1 else out[t]
            k = a * (0.6 + 0.8 * w[t])
            x = x + k * (src - x)
            buf[t] = x
        out = buf
    return out


def tongue_track(phone_ids: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame unified phone ids -> (T, 5) tongue parameters."""
    return track_from_targets(TARGETS[phone_ids], CRIT[phone_ids], fps)


def tongue_from_posteriors(post: np.ndarray, fps: float) -> np.ndarray:
    """Phone posteriors (T, N_PHONES) from the audio model -> (T, 5) tongue parameters."""
    return track_from_targets(post @ TARGETS, post @ CRIT, fps)


# --------------------------------------------------------------------------- geometry
CENTRE = np.array([0.0, -4.85, 2.95])
RADII = np.array([1.45, 0.45, 1.55])
_BASE = None


def _base():
    global _BASE
    if _BASE is None:
        from mimetic.face.render import uv_sphere
        v, f = uv_sphere(1.0, 28, 14)
        _BASE = (v, f)
    return _BASE


def tongue_mesh(params: np.ndarray):
    """(5,) params -> vertices (V,3) in the mandible frame (rest jaw), faces (F,3)."""
    v, f = _base()
    tip_up, tip_fwd, front_up, back_up, low = [float(x) for x in params]
    s = (v[:, 2] + 1) / 2                      # 0 = root, 1 = tip
    top = np.clip(v[:, 1] + 0.3, 0, 1.3) / 1.3  # upper surface moves most
    tip_w = np.clip((s - 0.62) / 0.38, 0, 1) ** 1.6
    front_w = np.exp(-0.5 * ((s - 0.62) / 0.16) ** 2)
    back_w = np.exp(-0.5 * ((s - 0.25) / 0.17) ** 2)
    x = v * RADII + CENTRE
    narrow = 1 - 0.35 * tip_w * min(1.0, max(tip_up, tip_fwd, 0) / 0.6)   # a raised tip narrows
    x[:, 0] *= narrow
    dy = tip_up * tip_w + front_up * front_w * top + back_up * back_w * top - low * (0.5 + 0.5 * s)
    dz = tip_fwd * tip_w
    x[:, 1] += dy
    x[:, 2] += dz
    return x.astype(np.float32), f
