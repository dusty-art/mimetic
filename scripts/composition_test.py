"""Speech under emotion: do lips still round for "oo" and seal for "m" while smiling or angry?

For every system and emotion, over TTS-driven test sentences:
  rounding   mouth width at unrounded vowels minus width at rounded sounds (mm); speech needs > 0
  closure    share of p/b/m occurrences whose lips seal (min aperture < 1 mm)
  smile      mouth-corner lift during the sentence vs neutral (mm); emotion needs > 0 for happy
  cheek      cheek raise (mm), the part of a smile that should survive rounding
Ratios vs neutral show whether an emotion steals the articulators or shares them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mimetic.data import motion as MO  # noqa: E402
from mimetic.data import phonemes as P  # noqa: E402
from mimetic.data import prosody  # noqa: E402
from mimetic.model.articulation import measures  # noqa: E402
from mimetic.model.controller import EMOTIONS  # noqa: E402
from mimetic.paths import EXPERIMENTS  # noqa: E402
from mimetic.runtime.engine import emotion_track, phone_track, resample_audio  # noqa: E402
from mimetic.runtime.face_model import load_face_model  # noqa: E402
from mimetic.runtime.tts import TextToSpeech  # noqa: E402
from evaluate import gauss_smooth, load_ctrl, viseme_tables  # noqa: E402
from train_controller import Data  # noqa: E402

SENTENCES = ["Who knew you would choose blue shoes?", "We were wondering where you went.",
             "My mom made a marvelous pie.", "Bob bought two boats by the pool.",
             "Would you move the food to the room?", "Please pass me the purple paper.",
             "Sue threw a few smooth stones.", "Maybe we will meet by the moon."]
ROUND = {"UW", "W", "OW", "UH", "SH", "ZH", "CH", "JH"}
UNROUND = {"IY", "IH", "EH", "AE", "EY", "AY"}
BIL = {"P", "B", "M"}


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    face = load_face_model(dev)
    rest = face.rest
    pca = np.load(MO.MOTION / "pca.npz")
    comps, mean = torch.tensor(pca["comps"], device=dev), torch.tensor(pca["mean"], device=dev)
    tts = TextToSpeech()
    utts = [tts.synthesize(s) for s in SENTENCES]
    systems = {k: load_ctrl(v, dev) for k, v in (("mimetic", "controller"), ("direct", "baseline_direct"),
                                                   ("noartic", "controller_noartic"))
               if (Path("checkpoints") / f"{v}.pt").exists()}
    vis, emo_off = viseme_tables(Data("train"))
    out = {}
    for emo in ("neutral", "happy", "angry", "sad", "surprised"):
        inten = 0.0 if emo == "neutral" else 1.0
        for sysname in list(systems) + ["viseme_table"]:
            rec = {"round": [], "close_hit": 0, "close_tot": 0, "corner": [], "cheek": []}
            for u in utts:
                n = int(np.ceil(u.duration * 30)) + 3
                ids, pos, dur = phone_track(u.segments, n)
                a16 = resample_audio(u.audio, 24000)
                pr = prosody.features(a16, fps=30, n=n)
                e, it = emotion_track([(0.0, u.duration, emo, inten)], n)
                t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)[None]
                with torch.no_grad():
                    if sysname == "viseme_table":
                        seq = vis[ids] + emo_off[EMOTIONS.index(emo)] * inten
                        x = rest + (torch.tensor(gauss_smooth(seq, 1.5), device=dev, dtype=torch.float32) @ comps + mean).view(n, -1, 3)
                    else:
                        model, direct, ck = systems[sysname]
                        o, _ = model(t(ids, torch.long), t(pos), t(dur), t(pr), t(e, torch.long), t(it),
                                     torch.tensor([ck["n_speakers"]], device=dev))
                        if direct:
                            x = rest + ((o[0] * torch.as_tensor(ck["pstd"], device=dev)) @ comps + mean).view(n, -1, 3)
                        else:
                            x = face(o[0])
                m = measures(x)
                w, ap = m["width"].cpu().numpy() * 10, m["aperture"].cpu().numpy() * 10
                centre = (np.abs(pos - 0.5) < 0.3)
                names = np.array(P.PHONES)[ids]
                r = np.isin(names, list(ROUND)) & centre
                ur = np.isin(names, list(UNROUND)) & centre
                if r.any() and ur.any():
                    rec["round"].append(w[ur].mean() - w[r].mean())
                tt = 0
                isb = np.isin(names, list(BIL))
                while tt < n:
                    if isb[tt]:
                        s0 = tt
                        while tt < n and isb[tt]:
                            tt += 1
                        rec["close_tot"] += 1
                        rec["close_hit"] += int(ap[max(0, s0 - 1):tt + 1].min() < 1.0)
                    else:
                        tt += 1
                d = (x - rest).cpu().numpy() * 10
                rec["corner"].append(((d[:, 61, 1] + d[:, 291, 1]) / 2).mean())
                rec["cheek"].append(((d[:, 50, 1] + d[:, 280, 1]) / 2).mean())
            out.setdefault(sysname, {})[emo] = {"rounding_mm": float(np.mean(rec["round"])),
                                                 "closure": rec["close_hit"] / max(1, rec["close_tot"]),
                                                 "corner_lift_mm": float(np.mean(rec["corner"])),
                                                 "cheek_raise_mm": float(np.mean(rec["cheek"]))}
    for s, d in out.items():
        base = d["neutral"]["rounding_mm"]
        for e, v in d.items():
            v["rounding_vs_neutral"] = v["rounding_mm"] / base if abs(base) > 1e-6 else None
    (EXPERIMENTS / "composition.json").write_text(json.dumps(out, indent=1))
    print(f"{'system':14s} {'emotion':10s} {'round mm':>9s} {'r/neutral':>9s} {'closure':>8s} {'corner mm':>9s} {'cheek mm':>9s}")
    for s, d in out.items():
        for e, v in d.items():
            print(f"{s:14s} {e:10s} {v['rounding_mm']:9.2f} {v['rounding_vs_neutral'] or 0:9.2f} {v['closure']:8.2f} "
                  f"{v['corner_lift_mm']:9.2f} {v['cheek_raise_mm']:9.2f}")


if __name__ == "__main__":
    main()
