"""Assemble the controller training set at 30 fps.

Per frame: fitted muscle activations, phoneme id / position / duration, prosody,
head rotation, blink mask. Per clip: emotion, intensity, speaker, dataset.
Output: data/ctrl/dataset.npz (+ clips.json).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from mimetic.data import motion as MO  # noqa: E402
from mimetic.data import phonemes as P  # noqa: E402
from mimetic.data import prosody, track  # noqa: E402
from mimetic.data.sources import EMOTIONS  # noqa: E402
from mimetic.paths import DATA  # noqa: E402

FPS = 30.0
CTRL = DATA / "ctrl"


def resample(x, n_out, fps_in):
    t_in = np.arange(len(x)) / fps_in
    t_out = np.arange(n_out) / FPS
    flat = x.reshape(len(x), -1).astype(np.float32)
    out = np.stack([np.interp(t_out, t_in, flat[:, k]) for k in range(flat.shape[1])], -1)
    return out.reshape(n_out, *x.shape[1:])


def phone_frames(segs, n):
    """-> ids (n,), position-in-phone (n,), phone duration s (n,), word index (n,)"""
    ids = np.zeros(n, np.int64)
    pos = np.zeros(n, np.float32)
    dur = np.zeros(n, np.float32)
    word = np.full(n, -1, np.int64)
    for a, b, p, w in segs:
        i0, i1 = int(round(a * FPS)), max(int(round(a * FPS)) + 1, int(round(b * FPS)))
        i0, i1 = max(0, i0), min(n, i1)
        if i1 <= i0:
            continue
        ids[i0:i1] = P.PID[p]
        pos[i0:i1] = (np.arange(i0, i1) + 0.5 - a * FPS) / max((b - a) * FPS, 1e-3)
        dur[i0:i1] = b - a
        word[i0:i1] = w
    return ids, np.clip(pos, 0, 1), dur, word


def build_speaker(name):
    z = np.load(MO.MOTION / f"{name}.npz")
    acts_path = MO.MOTION / f"{name}_acts.npy"
    if not acts_path.exists():
        return None
    acts = np.load(acts_path).astype(np.float32)
    index = json.loads(z["index"].tobytes().decode())
    rot = z["rot"]
    coef = z["coef"].astype(np.float32)
    bs = z["bs"].astype(np.float32)
    bi = {n: i for i, n in enumerate(MO.BS_NAMES)}
    blink_raw = np.maximum(bs[:, bi["eyeBlinkLeft"]], bs[:, bi["eyeBlinkRight"]])
    recs = []
    for c in index:
        o, n, fps = c["offset"], c["n"], c["fps"]
        phn = Path(c["path"]).with_suffix(".phn.json")
        if not phn.exists():
            continue
        segs = json.loads(phn.read_text()).get("segments", [])
        if not segs:
            continue
        n30 = int(round(n / fps * FPS))
        a = resample(acts[o:o + n], n30, fps)
        r = resample(rot[o:o + n], n30, fps)
        cf = resample(coef[o:o + n], n30, fps)
        r = r - np.median(r, 0, keepdims=True)
        blink = resample(blink_raw[o:o + n], n30, fps) > 0.45
        d = track.load_clip(c["path"])
        pr = prosody.features(d["audio"], fps=FPS, n=n30) if len(d["audio"]) else np.zeros((n30, 4), np.float32)
        ids, pos, dur, word = phone_frames(segs, n30)
        recs.append({"acts": a.astype(np.float16), "rot": r.astype(np.float16), "blink": blink, "coef": cf.astype(np.float16),
                     "phone": ids.astype(np.int8), "ppos": pos.astype(np.float16), "pdur": dur.astype(np.float16),
                     "prosody": pr.astype(np.float16),
                     "meta": {"speaker": name, "clip": c["clip"], "emotion": c["emotion"],
                              "intensity": c["intensity"], "dataset": c["dataset"], "text": c["text"], "n": n30}})
    return recs


def fitted_face_coefs(acts):
    """PCA coefficients of the muscle-fit face for every frame, so training never re-decodes targets."""
    import torch
    from mimetic.runtime.face_model import load_face_model
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    face = load_face_model(dev)
    pca = np.load(MO.MOTION / "pca.npz")
    comps, mean = torch.tensor(pca["comps"], device=dev), torch.tensor(pca["mean"], device=dev)
    rest = face.rest.reshape(-1)
    out = np.zeros((len(acts), comps.shape[0]), np.float16)
    with torch.no_grad():
        for i in range(0, len(acts), 8192):
            a = torch.tensor(acts[i:i + 8192].astype(np.float32), device=dev)
            d = face(a).reshape(len(a), -1) - rest
            out[i:i + 8192] = ((d - mean) @ comps.T).cpu().numpy().astype(np.float16)
    return out


def main():
    CTRL.mkdir(parents=True, exist_ok=True)
    sp = np.load(MO.MOTION / "speakers.npz")
    names = [str(n) for n in sp["names"]]
    allrec = []
    with ProcessPoolExecutor(6) as ex:
        for recs in ex.map(build_speaker, names):
            if recs:
                allrec += recs
    keys = ["acts", "rot", "blink", "coef", "phone", "ppos", "pdur", "prosody"]
    out = {k: np.concatenate([r[k] for r in allrec]) for k in keys}
    offs = np.cumsum([0] + [r["meta"]["n"] for r in allrec])
    metas = [dict(r["meta"], offset=int(o)) for r, o in zip(allrec, offs[:-1])]
    out["fitcoef"] = fitted_face_coefs(out["acts"])
    np.savez(CTRL / "dataset.npz", **out)
    (CTRL / "clips.json").write_text(json.dumps({"clips": metas, "speakers": names, "emotions": EMOTIONS}))
    dsets = {}
    for m in metas:
        dsets[m["dataset"]] = dsets.get(m["dataset"], 0) + m["n"]
    print(f"{len(metas)} clips, {offs[-1]} frames ({offs[-1] / FPS / 3600:.1f} h) {dsets}")


if __name__ == "__main__":
    main()
