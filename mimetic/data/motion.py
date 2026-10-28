"""Motion database: every tracked clip as head-pose-free displacements from its speaker's neutral face.

Layout (data/motion/):
  speakers.npz         per-speaker neutral shapes, the population mean neutral
  pca.npz              displacement PCA basis (mean, components, explained variance)
  {speaker}.npz        per-frame PCA coefficients (float16), head pose, iris, blendshapes,
                       and a clip index (offsets, fps, emotion, intensity, text, dataset)

Displacements are stored as PCA coefficients: 96 numbers per frame instead of
1404 reconstruct the tracked lips to a fraction of a millimetre while keeping
the whole corpus under a gigabyte.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from mimetic.data import canon, track
from mimetic.face import canonical as C
from mimetic.paths import DATA, PROCESSED

MOTION = DATA / "motion"
BS_NAMES = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
            'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
            'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
            'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
            'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
            'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft',
            'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
            'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft',
            'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
            'noseSneerLeft', 'noseSneerRight']
K_PCA = 96


def speaker_dirs():
    return sorted(p for p in PROCESSED.glob("*/*") if p.is_dir() and not p.name.startswith("_"))


def _neutral_mask(bs, emotion):
    i = canon.bs_index(BS_NAMES)
    g = lambda n: bs[:, i[n]].astype(np.float32)
    calm = emotion in ("neutral", "calm")
    m = (g("jawOpen") < 0.09) & (g("eyeBlinkLeft") < 0.35) & (g("eyeBlinkRight") < 0.35) \
        & (g("mouthSmileLeft") < 0.3) & (g("mouthSmileRight") < 0.3) & (g("mouthPucker") < 0.4)
    return m & calm


def load_speaker(sdir: Path, max_clips: int | None = None):
    clips = []
    for f in sorted(sdir.glob("*.npz"))[:max_clips]:
        if f.name.endswith(".tmp.npz"):
            continue
        try:
            d = track.load_clip(f)
        except Exception:
            continue
        ok = d["ok"]
        if ok.sum() < 10:
            continue
        # Fill tracking dropouts by holding the previous good frame.
        first = int(np.argmax(ok))
        idx = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), first))
        d["lmk"] = d["lmk"][idx]
        d["bs"] = d["bs"][idx]
        d["path"] = str(f)
        clips.append(d)
    return clips


def speaker_neutral(clips):
    locs, masks = [], []
    for d in clips:
        loc, _, _ = canon.image_to_local(d["lmk"])
        locs.append(loc)
        masks.append(_neutral_mask(d["bs"], str(d["emotion"])))
    L = np.concatenate(locs)
    M = np.concatenate(masks)
    if M.sum() < 30:   # speakers without neutral clips: least-expressive frames overall
        i = canon.bs_index(BS_NAMES)
        score = np.concatenate([d["bs"][:, [i["jawOpen"], i["mouthSmileLeft"], i["mouthSmileRight"],
                                            i["browInnerUp"], i["eyeBlinkLeft"]]].astype(np.float32).sum(1)
                                for d in clips])
        M = score <= np.quantile(score, 0.05)
    N = np.median(L[M], 0)
    # Re-register the neutral onto the canonical frame with all vertices (removes residual pose bias).
    s, R, t = canon.procrustes(N[None], C.load_canonical().verts, np.ones(468))
    N = s[0] * N @ R[0].T + t[0]
    return N, int(M.sum())


def clip_displacements(d, N):
    loc, (s, R, t), iris = canon.image_to_local(d["lmk"], ref=N)
    return loc - N[None], (s, R, t), iris


def rotvec(R):
    """Batched rotation matrix -> axis-angle (T,3)."""
    cos = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    ang = np.arccos(cos)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 1e-8, v / np.maximum(n, 1e-8) * ang[:, None], 0.0)


# --------------------------------------------------------------------------- lip contact transfer
LIP_U = C.UPPER_INNER[1:-1]
LIP_L = C.LOWER_INNER[1:-1]


def pair_apertures(shapes):
    """(T,468,3) -> (T,9) vertical gap of each inner-lip pair (cm)."""
    return shapes[:, LIP_U, 1] - shapes[:, LIP_L, 1]


def lip_contact_params(locals_, N):
    """Per pair: the speaker's sealed gap (2nd percentile) and neutral gap."""
    ap = np.concatenate([pair_apertures(L) for L in locals_])
    closed = np.percentile(ap, 2, axis=0)
    neutral = pair_apertures(N[None])[0]
    return closed, neutral


_LIP_W = None


def _lip_weights(N):
    """Smooth spread of a pair's vertical correction over its lip (upper pairs -> upper lip only)."""
    global _LIP_W
    if _LIP_W is None:
        from mimetic.face import anatomy as A
        X = C.load_canonical().verts
        ym = A.mouth_line_y(C.load_canonical(), X[:, 0])
        Wu = np.zeros((468, len(LIP_U)))
        Wl = np.zeros((468, len(LIP_L)))
        lipreg = (np.abs(X[:, 0]) < 3.0) & (np.abs(X[:, 1] - ym) < 1.3)
        for k, (u, l) in enumerate(zip(LIP_U, LIP_L)):
            du = np.linalg.norm(X - X[u], axis=1)
            dl = np.linalg.norm(X - X[l], axis=1)
            Wu[:, k] = np.exp(-0.5 * (du / 0.35) ** 2) * (X[:, 1] > ym) * lipreg
            Wl[:, k] = np.exp(-0.5 * (dl / 0.35) ** 2) * (X[:, 1] <= ym) * lipreg
        Wu /= np.maximum(Wu.sum(1, keepdims=True), 1.0)
        Wl /= np.maximum(Wl.sum(1, keepdims=True), 1.0)
        for k, (u, l) in enumerate(zip(LIP_U, LIP_L)):
            Wu[u] = 0; Wu[u, k] = 1
            Wl[l] = 0; Wl[l, k] = 1
        _LIP_W = (Wu, Wl)
    return _LIP_W


def apply_lip_contact(D, N, closed, neutral, rest_gap):
    """Correct transferred displacements so lip contact transfers between faces.

    A speaker's sealed lips map to touching lips on the rest face and their
    neutral gap maps to the rest face's neutral gap. Without this, a speaker
    whose relaxed lips sit further apart than average drives the rest face's
    lips through each other whenever they seal.
    """
    local_ap = pair_apertures(N[None] + D)                       # speaker's own gap per frame
    transferred = rest_gap[None] + (local_ap - neutral[None])
    span = np.maximum(neutral - closed, 0.05)
    below = rest_gap[None] * np.clip(local_ap - closed[None], 0, None) / span[None]
    want = np.where(local_ap < neutral[None], below, transferred)
    delta = want - transferred                                    # (T,9) + opens, - closes
    Wu, Wl = _lip_weights(N)
    D = D.copy()
    D[:, :, 1] += 0.5 * delta @ Wu.T - 0.5 * delta @ Wl.T
    return D


def pair_apertures_t(x):
    """Torch version of pair_apertures for (..., 468, 3)."""
    return x[..., LIP_U, 1] - x[..., LIP_L, 1]


# --------------------------------------------------------------------------- eyelid transfer
LID_U = C.R_UPPER_LID[1:-1] + C.L_UPPER_LID[1:-1]
LID_L = C.R_LOWER_LID[1:-1] + C.L_LOWER_LID[1:-1]


def lid_apertures(shapes):
    """(T,468,3) -> (T,14) vertical gap of each eyelid pair (cm)."""
    return shapes[..., LID_U, 1] - shapes[..., LID_L, 1]


def lid_apertures_t(x):
    return x[..., LID_U, 1] - x[..., LID_L, 1]


_LID_W = None


def _lid_weights():
    global _LID_W
    if _LID_W is None:
        X = C.load_canonical().verts
        Wu = np.zeros((468, len(LID_U)))
        Wl = np.zeros((468, len(LID_L)))
        for k, (u, l) in enumerate(zip(LID_U, LID_L)):
            side = np.sign(X[u, 0])
            same = np.sign(X[:, 0]) == side
            eye_y = (X[u, 1] + X[l, 1]) / 2
            du = np.linalg.norm(X - X[u], axis=1)
            dl = np.linalg.norm(X - X[l], axis=1)
            Wu[:, k] = np.exp(-0.5 * (du / 0.45) ** 2) * same * (X[:, 1] > eye_y)
            Wl[:, k] = np.exp(-0.5 * (dl / 0.35) ** 2) * same * (X[:, 1] <= eye_y)
        Wu /= np.maximum(Wu.sum(1, keepdims=True), 1.0)
        Wl /= np.maximum(Wl.sum(1, keepdims=True), 1.0)
        for k, (u, l) in enumerate(zip(LID_U, LID_L)):
            Wu[u] = 0; Wu[u, k] = 1
            Wl[l] = 0; Wl[l, k] = 1
        _LID_W = (Wu, Wl)
    return _LID_W


def apply_lid_transfer(D, N, rest):
    """Eye openness transfers as a ratio: a speaker whose eyes are half closed shows half-closed eyes on
    the rest face, whatever the two faces' natural eye sizes (an offset would close small eyes too far
    and large eyes too little). The upper lid carries 75% of the change, as in real lid motion."""
    neutral = lid_apertures(N[None])[0]
    rest_gap = lid_apertures(rest[None])[0]
    local = lid_apertures(N[None] + D)
    transferred = rest_gap[None] + (local - neutral[None])
    want = rest_gap[None] * np.clip(local / np.maximum(neutral[None], 0.05), 0, 2.0)
    delta = want - transferred
    Wu, Wl = _lid_weights()
    D = D.copy()
    D[:, :, 1] += 0.75 * delta @ Wu.T - 0.25 * delta @ Wl.T
    return D
