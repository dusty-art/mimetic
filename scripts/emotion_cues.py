"""How much of each emotion's facial signature survives: real faces vs muscle fit vs controllers.

Per emotion, the mean change from neutral (mm) of interpretable cues on held-out speakers.
Writes experiments/emotion_cues.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluate import geo_features, load_ctrl  # noqa: E402
from mimetic.data import motion as MO  # noqa: E402
from mimetic.face import anatomy as A  # noqa: E402
from mimetic.paths import CHECKPOINTS, EXPERIMENTS  # noqa: E402
from mimetic.runtime.face_model import load_face_model  # noqa: E402
from train_controller import Data, to_dev  # noqa: E402

FEATS = ["aperture", "width", "protr", "corner", "ulip", "llip", "brow_mid", "brow_inner", "brow_together",
         "eye_open", "cheek", "nostril", "chin", "asym"]
CUES = {"happy": ["corner", "cheek", "eye_open"], "sad": ["brow_inner", "corner", "chin"],
        "angry": ["brow_inner", "brow_together", "eye_open"], "fearful": ["eye_open", "brow_inner", "width"],
        "disgusted": ["nostril", "brow_inner", "eye_open"], "surprised": ["eye_open", "brow_inner", "aperture"]}


def main():
    dev = "cpu"
    test = Data("test")
    face = load_face_model(dev)
    rest = face.rest
    pca = np.load(MO.MOTION / "pca.npz")
    comps, mean = torch.tensor(pca["comps"]), torch.tensor(pca["mean"])
    clips = [c for c in test.clips if c["dataset"] in ("ravdess", "cremad")]
    systems = {"real": None, "muscle_fit": None}
    for k, v in (("mimetic", "controller"), ("direct", "baseline_direct"), ("noartic", "controller_noartic")):
        if (CHECKPOINTS / f"{v}.pt").exists():
            systems[k] = load_ctrl(v, dev)
    feats = {k: {} for k in systems}
    with torch.no_grad():
        for j in range(0, len(clips), 64):
            cl = clips[j:j + 64]
            b = to_dev(test.collate(cl, None), dev)
            m = b["mask"]
            for k, s in systems.items():
                if k == "real":
                    x = rest + (b["coef"] @ comps + mean).view(*m.shape, -1, 3)
                elif k == "muscle_fit":
                    x = face(b["acts"].reshape(-1, A.N_ACT)).view(*m.shape, -1, 3)
                else:
                    model, direct, ck = s
                    o, _ = model(b["phone"], b["ppos"], b["pdur"], b["prosody"], b["emo"], b["inten"],
                                 torch.full_like(b["spk"], ck["n_speakers"]), pad_mask=m == 0)
                    x = rest + ((o * torch.as_tensor(ck["pstd"])) @ comps + mean).view(*m.shape, -1, 3) if direct \
                        else face(o.reshape(-1, A.N_ACT)).view(*m.shape, -1, 3)
                f = (geo_features(x, rest) * m[..., None]).sum(1) / m.sum(1, keepdim=True)
                for i, c in enumerate(cl):
                    feats[k].setdefault(c["emotion"], []).append(f[i].numpy())
    out = {}
    for k, d in feats.items():
        base = np.mean(d["neutral"], 0)
        out[k] = {e: {cue: float(np.mean(d[e], 0)[FEATS.index(cue)] - base[FEATS.index(cue)]) for cue in cues}
                  for e, cues in CUES.items() if e in d}
    # Share of the real signature reproduced (signed projection, averaged over each emotion's cues)
    for k in out:
        if k == "real":
            continue
        shares = []
        for e in out[k]:
            r = np.array([out["real"][e][c] for c in CUES[e]])
            p = np.array([out[k][e][c] for c in CUES[e]])
            shares.append(float(np.dot(p, r) / max(np.dot(r, r), 1e-6)))
        out[k]["_signature_share"] = float(np.mean(shares))
    (EXPERIMENTS / "emotion_cues.json").write_text(json.dumps(out, indent=1))
    for k in out:
        print(k, {e: {c: round(v, 2) for c, v in cv.items()} for e, cv in out[k].items() if not e.startswith("_")},
              "share", round(out[k].get("_signature_share", 1.0), 2))


if __name__ == "__main__":
    main()
