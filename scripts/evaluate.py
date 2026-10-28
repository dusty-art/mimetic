"""Evaluate talking-face systems on held-out speakers against tracked ground truth.

Systems: mimetic (muscle controller), ablations, direct-regression baseline (same
network, no muscle layer), viseme-table baseline (classic lookup + smoothing +
additive emotion offsets), and the muscle fit itself (representation ceiling).

Metrics (canonical face space):
  LVE      lip vertex error: per-frame max error over lip vertices, mean over frames (mm)
  MVE      mean vertex error over the face (mm)
  ApCorr   Pearson correlation of lip aperture with ground truth (mean over clips)
  Closure  share of p/b/m occurrences where the lips close (min aperture < 1 mm)
  EmoAcc   emotion accuracy of a classifier trained on real faces, applied to the system's faces
  Jerk     mean squared third difference of lip motion relative to ground truth (1.0 = as smooth as real)
  Pen      frames with lips interpenetrating (aperture < -0.5 mm), %
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from mimetic.data import motion as MO  # noqa: E402
from mimetic.data import phonemes as P  # noqa: E402
from mimetic.face import anatomy as A  # noqa: E402
from mimetic.face import canonical as C  # noqa: E402
from mimetic.model.articulation import measures  # noqa: E402
from mimetic.model.controller import EMOTIONS, SpeechEmotionController  # noqa: E402
from mimetic.paths import CHECKPOINTS, EXPERIMENTS  # noqa: E402
from mimetic.runtime.face_model import load_face_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_controller import Data, to_dev  # noqa: E402

LIPS = C.UPPER_INNER + C.LOWER_INNER + C.UPPER_OUTER + C.LOWER_OUTER


class EmoClassifier(nn.Module):
    def __init__(self, k=96, n=len(EMOTIONS)):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(k, 128, 5, padding=2), nn.GELU(), nn.Conv1d(128, 128, 5, padding=4, dilation=2),
                                 nn.GELU(), nn.Conv1d(128, 128, 5, padding=8, dilation=4), nn.GELU())
        self.out = nn.Linear(128, n)

    def forward(self, c, m):
        h = self.net(c.transpose(1, 2)).transpose(1, 2)
        h = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        return self.out(h)


def coef_of(x, rest, comps, mean, pstd):
    d = (x - rest).reshape(*x.shape[:-2], -1)
    return ((d - mean) @ comps.T) / pstd


def train_emotion_classifier(train, comps, mean, pstd, dev, steps=3000):
    clf = EmoClassifier(comps.shape[0]).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(1)
    emo_clips = [i for i, c in enumerate(train.clips) if c["dataset"] in ("ravdess", "cremad")]
    for s in range(steps):
        idx = rng.choice(emo_clips, 64)
        b = to_dev(train.collate([train.clips[i] for i in idx], rng), dev)
        c = b["coef"] / pstd
        c = c + 0.05 * torch.randn_like(c)
        loss = nn.functional.cross_entropy(clf(c, b["mask"]), b["emo"])
        opt.zero_grad(); loss.backward(); opt.step()
    return clf.eval()


def viseme_tables(train):
    """Per-phoneme mean displacement and per-emotion mean offset (classic lookup baseline)."""
    ph, cf = train.arr["phone"], train.arr["coef"].astype(np.float32)
    K = cf.shape[1]
    vis = np.zeros((P.N_PHONES, K), np.float32)
    for p in range(P.N_PHONES):
        sel = ph == p
        if sel.sum() > 50:
            vis[p] = cf[sel].mean(0)
    emo = np.zeros((len(EMOTIONS), K), np.float32)
    neu = [c for c in train.clips if c["emotion"] == "neutral" and c["dataset"] != "grid"]
    base = np.concatenate([cf[c["offset"]:c["offset"] + c["n"]] for c in neu]).mean(0) if neu else 0
    for e in range(len(EMOTIONS)):
        cs = [c for c in train.clips if c["emo"] == e and c["inten"] > 0]
        if cs:
            emo[e] = np.concatenate([cf[c["offset"]:c["offset"] + c["n"]] for c in cs]).mean(0) - base
    return vis, emo


def gauss_smooth(x, sigma):
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], k, "valid") for j in range(x.shape[1])], -1)


def load_ctrl(name, dev, n_act=None):
    ck = torch.load(CHECKPOINTS / f"{name}.pt", map_location="cpu", weights_only=False)
    direct = ck["args"].get("baseline") == "direct"
    m = SpeechEmotionController(n_speakers=ck["n_speakers"], n_act=96 if direct else A.N_ACT,
                                output="linear" if direct else "sigmoid")
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), direct, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="mimetic=controller,direct=baseline_direct,noartic=controller_noartic")
    ap.add_argument("--max-clips", type=int, default=100000)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    train, test = Data("train"), Data("test")
    face = load_face_model(dev)
    rest = face.rest
    pca = np.load(MO.MOTION / "pca.npz")
    comps, mean = torch.tensor(pca["comps"], device=dev), torch.tensor(pca["mean"], device=dev)
    pstd = torch.tensor(train.arr["coef"][::50].astype(np.float32).std(0), device=dev).clamp_min(1e-3)
    print("training emotion classifier on real faces...", flush=True)
    clf = train_emotion_classifier(train, comps, mean, pstd, dev)
    vis, emo_off = viseme_tables(train)
    systems = {}
    for kv in args.systems.split(","):
        k, v = kv.split("=")
        if (CHECKPOINTS / f"{v}.pt").exists():
            systems[k] = load_ctrl(v, dev)
    rest_ap = measures(rest[None])["aperture"].item()
    close_thr = 0.10
    clips = test.clips[: args.max_clips]
    names = ["ground_truth", "muscle_fit", "viseme_table"] + list(systems)
    acc = {n: {"lve": [], "mve": [], "apcorr": [], "close_hit": 0, "close_tot": 0, "emo_hit": 0, "emo_tot": 0,
               "jerk": [], "pen": [], "frames": 0} for n in names}
    t_run = {n: 0.0 for n in names}
    bil = torch.tensor([P.PID[p] for p in ("P", "B", "M")], device=dev)
    emo_ids = set(i for i, e in enumerate(EMOTIONS))
    with torch.no_grad():
        for j in range(0, len(clips), 32):
            cl = clips[j:j + 32]
            b = to_dev(test.collate(cl, None), dev)
            m = b["mask"]
            gt = rest + ((b["coef"]) @ comps + mean).view(*m.shape, -1, 3)
            preds = {"ground_truth": gt, "muscle_fit": face(b["acts"].reshape(-1, A.N_ACT)).view(*m.shape, -1, 3)}
            # viseme baseline
            vb = np.zeros((len(cl), m.shape[1], comps.shape[0]), np.float32)
            ph = b["phone"].cpu().numpy()
            for i, c in enumerate(cl):
                n = c["n"]
                seq = vis[ph[i, :n]] + emo_off[c["emo"]] * c["inten"]
                vb[i, :n] = gauss_smooth(seq, 1.5)
            preds["viseme_table"] = rest + (torch.tensor(vb, device=dev) @ comps + mean).view(*m.shape, -1, 3)
            spk = torch.full_like(b["spk"], 0)
            for k, (model, direct, ck) in systems.items():
                t0 = time.time()
                spk = torch.full_like(b["spk"], ck["n_speakers"])          # average style for unseen speakers
                out, _ = model(b["phone"], b["ppos"], b["pdur"], b["prosody"], b["emo"], b["inten"], spk,
                               pad_mask=m == 0)
                if direct:
                    x = rest + ((out * torch.as_tensor(ck["pstd"], device=dev)) @ comps + mean).view(*m.shape, -1, 3)
                else:
                    x = face(out.reshape(-1, A.N_ACT)).view(*m.shape, -1, 3)
                if dev == "mps":
                    torch.mps.synchronize()
                t_run[k] += time.time() - t0
                preds[k] = x
            for k, x in preds.items():
                a_ = acc[k]
                err = (x - gt).norm(dim=-1)                                     # (B,T,N)
                lve = err[..., LIPS].max(-1).values
                a_["lve"].append(((lve * m).sum() / m.sum()).item() * 10)
                a_["mve"].append(((err.mean(-1) * m).sum() / m.sum()).item() * 10)
                ap_p, ap_g = measures(x)["aperture"], measures(gt)["aperture"]
                for i, c in enumerate(cl):
                    n = c["n"]
                    pp, gg = ap_p[i, :n].cpu().numpy(), ap_g[i, :n].cpu().numpy()
                    if pp.std() > 1e-6 and gg.std() > 1e-6:
                        a_["apcorr"].append(np.corrcoef(pp, gg)[0, 1])
                    # closure on bilabial runs
                    isb = torch.isin(b["phone"][i, :n], bil).cpu().numpy()
                    t = 0
                    while t < n:
                        if isb[t]:
                            s0 = t
                            while t < n and isb[t]:
                                t += 1
                            a_["close_tot"] += 1
                            a_["close_hit"] += int(pp[max(0, s0 - 1):t + 1].min() < close_thr)
                        else:
                            t += 1
                    a_["pen"].append(float((pp < -0.05).mean()))
                    lipm = x[i, :n][:, LIPS]
                    if n > 4:
                        jk = lipm[3:] - 3 * lipm[2:-1] + 3 * lipm[1:-2] - lipm[:-3]
                        a_["jerk"].append((jk ** 2).sum(-1).mean().item())
                a_["frames"] += int(m.sum().item())
                # emotion recognisability (emotional datasets only)
                sel = torch.tensor([c["dataset"] in ("ravdess", "cremad") for c in cl], device=dev)
                if sel.any():
                    logits = clf(coef_of(x, rest, comps, mean, pstd)[sel], m[sel])
                    a_["emo_hit"] += int((logits.argmax(-1) == b["emo"][sel]).sum().item())
                    a_["emo_tot"] += int(sel.sum().item())
    gt_jerk = np.mean(acc["ground_truth"]["jerk"])
    res = {}
    for k, a_ in acc.items():
        res[k] = {"LVE_mm": float(np.mean(a_["lve"])), "MVE_mm": float(np.mean(a_["mve"])),
                  "ApCorr": float(np.mean(a_["apcorr"])), "Closure": a_["close_hit"] / max(1, a_["close_tot"]),
                  "EmoAcc": a_["emo_hit"] / max(1, a_["emo_tot"]), "Jerk_rel": float(np.mean(a_["jerk"]) / gt_jerk),
                  "Pen_pct": 100 * float(np.mean(a_["pen"])),
                  "RTF": (t_run[k] / (a_["frames"] / 30.0)) if t_run[k] else None}
    res["_meta"] = {"test_clips": len(clips), "closure_threshold_cm": close_thr, "rest_aperture_cm": rest_ap,
                    "bilabial_occurrences": acc["ground_truth"]["close_tot"]}
    EXPERIMENTS.mkdir(exist_ok=True)
    (EXPERIMENTS / "eval_results.json").write_text(json.dumps(res, indent=1))
    hdr = f"{'system':14s} {'LVE mm':>7s} {'MVE mm':>7s} {'ApCorr':>7s} {'Closure':>8s} {'EmoAcc':>7s} {'Jerk':>6s} {'Pen%':>5s}"
    print(hdr)
    for k, r in res.items():
        if k.startswith("_"):
            continue
        print(f"{k:14s} {r['LVE_mm']:7.2f} {r['MVE_mm']:7.2f} {r['ApCorr']:7.3f} {r['Closure']:8.3f} {r['EmoAcc']:7.3f} "
              f"{r['Jerk_rel']:6.2f} {r['Pen_pct']:5.2f}")
    print(res["_meta"])


if __name__ == "__main__":
    main()
