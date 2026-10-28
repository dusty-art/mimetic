"""Muscle motion capture from any face video: track -> register -> contact-preserving transfer -> muscle encoder.

The result is an Animation like the text/audio paths produce, so it exports to
ARKit, Live Link, Blender and video the same way, retargeted through muscle
space and therefore physically valid on the target face.
"""
from __future__ import annotations

import numpy as np
import torch

from mimetic.data import canon, motion as MO, track
from mimetic.fit.encoder import TemporalEncoder
from mimetic.paths import ASSETS, CHECKPOINTS
from mimetic.runtime.engine import Animation


def video_to_animation(engine, path, device="cpu") -> Animation:
    tr = track.track_video(path)
    ok = tr["ok"]
    if ok.mean() < 0.5:
        raise RuntimeError("face not found in most frames")
    first = int(np.argmax(ok))
    idx = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), first))
    lmk = tr["lmk"][idx].astype(np.float32) * np.array([tr["width"], tr["height"], tr["width"]], np.float32)
    bs = tr["bs"][idx].astype(np.float32)
    loc, _, _ = canon.image_to_local(lmk)
    i = canon.bs_index(MO.BS_NAMES)
    score = bs[:, [i["jawOpen"], i["mouthSmileLeft"], i["mouthSmileRight"], i["browInnerUp"], i["eyeBlinkLeft"]]].sum(1)
    calm = score <= np.quantile(score, 0.15)
    N = np.median(loc[calm], 0)
    from mimetic.face import canonical as C
    s, R, t = canon.procrustes(N[None], C.load_canonical().verts, np.ones(468))
    N = s[0] * N @ R[0].T + t[0]
    loc, (s, R, t), _ = canon.image_to_local(lmk, ref=N)
    D = loc - N[None]
    rest = np.load(ASSETS / "rest_neutral.npy")
    closed, neutral = MO.lip_contact_params([loc], N)
    D = MO.apply_lip_contact(D, N, closed, neutral, MO.pair_apertures(rest[None])[0])
    D = MO.apply_lid_transfer(D, N, rest)
    pca = np.load(MO.MOTION / "pca.npz")
    coef = (D.reshape(len(D), -1) - pca["mean"]) @ pca["comps"].T
    ck = torch.load(CHECKPOINTS / "encoder.pt", map_location="cpu", weights_only=False)
    enc = TemporalEncoder(k_in=ck["k_in"])
    enc.load_state_dict(ck["enc"])
    enc = enc.to(device).eval()
    with torch.no_grad():
        acts = enc(torch.tensor((coef / ck["pca_std"])[None], dtype=torch.float32, device=device))[0].cpu().numpy()
    rv = MO.rotvec(np.transpose(R, (0, 2, 1)))
    head = (rv - np.median(rv, 0))[:, [0, 1, 2]].astype(np.float32)
    head[:, 0] *= -1
    audio = track.read_audio(path).astype(np.float32) / 32768.0
    fps = float(tr["fps"])
    return Animation(fps, acts.astype(np.float32), head, np.zeros((len(acts), 2), np.float32),
                     audio if len(audio) else None, 16000, [], [], engine)
