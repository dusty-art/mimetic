"""Tracked image-space landmarks -> head-pose-free local face shapes in canonical centimetres.

1. Flip to a right-handed y-up, z-toward-camera frame.
2. Weighted similarity Procrustes on vertices that sit on bone (forehead, nasal
   bridge, temples), so expressions do not leak into the head pose estimate.
3. Per speaker, a neutral shape is the median of low-expression frames; frames
   are re-registered to it and expressed as displacements from it.
"""
from __future__ import annotations

import numpy as np

from mimetic.face import canonical as C

BS = None  # blendshape name -> index, filled lazily


def anchor_weights(n=468):
    w = np.full(n, 0.02)
    w[C.RIGID_ANCHORS] = 1.0
    return w


def procrustes(P: np.ndarray, X: np.ndarray, w: np.ndarray):
    """Batched weighted similarity transform mapping P (T,N,3) onto X (N,3).

    Returns s (T,), R (T,3,3), t (T,3) with  s * P @ R^T + t ~= X.
    """
    w = w / w.sum()
    muP = np.einsum("n,tnc->tc", w, P)
    muX = w @ X
    Pc = P - muP[:, None]
    Xc = X - muX
    H = np.einsum("n,tni,nj->tij", w, Pc, Xc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", U, Vt)))
    D = np.ones_like(S)
    D[:, -1] = d
    R = np.einsum("tji,tj,tkj->tik", Vt, D, U)          # V diag(D) U^T
    varP = np.einsum("n,tnc->t", w, Pc ** 2)
    s = (S * D).sum(1) / varP
    t = muX - s[:, None] * np.einsum("tij,tj->ti", R, muP)
    return s, R, t


def image_to_local(lmk_px: np.ndarray, ref: np.ndarray | None = None, w: np.ndarray | None = None):
    """lmk_px (T,478,3) -> local (T,468,3) cm, plus pose (s, R, t) and iris (T,10,3) in the same frame."""
    P = lmk_px[:, :, :].astype(np.float64) * np.array([1.0, -1.0, -1.0])
    ref = C.load_canonical().verts if ref is None else ref
    w = anchor_weights() if w is None else w
    s, R, t = procrustes(P[:, :468], ref, w)
    allp = s[:, None, None] * np.einsum("tij,tnj->tni", R, P) + t[:, None]
    return allp[:, :468], (s, R, t), allp[:, 468:]


def bs_index(names):
    return {n: i for i, n in enumerate(names)}


def neutral_frames(bs: np.ndarray, names: list[str]) -> np.ndarray:
    """Boolean mask of low-expression frames from MediaPipe blendshape scores."""
    i = bs_index(names)
    g = lambda n: bs[:, i[n]].astype(np.float32)
    m = (g("jawOpen") < 0.06) & (g("mouthSmileLeft") < 0.2) & (g("mouthSmileRight") < 0.2) \
        & (g("browInnerUp") < 0.3) & (g("eyeBlinkLeft") < 0.4) & (g("eyeBlinkRight") < 0.4) \
        & (g("mouthPucker") < 0.35) & (g("mouthFunnel") < 0.2) & (g("browDownLeft") < 0.3) \
        & (g("browDownRight") < 0.3)
    return m
