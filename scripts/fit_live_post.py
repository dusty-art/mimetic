"""Fit the live post-processor (mimetic/runtime/live.py: LivePost) on training speakers only.

1. Emotion offsets: per-emotion mean muscle logits relative to the same speaker's neutral clips, from
   the RAVDESS / CREMA-D muscle mocap, averaged over speakers. Lip sphincter and jaw are zeroed so the
   offsets never fight articulation.
2. One Euro filter: sweep (min_cutoff, beta) on training-speaker clips and keep the setting with the
   best lip-aperture correlation among those whose lip jerk is no higher than real tracked faces and
   whose audio-visual offset stays within one frame of the unfiltered model.
3. Emotion pathway: sweep posterior smoothing, sharpening and gain on emotional training-speaker clips,
   scored by the same geometric emotion judge as evaluate.py; keep the most recognizable setting whose
   p/b/m closure stays within 3 points of the filtered model without emotion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mimetic.data import motion as MO  # noqa: E402
from mimetic.data import phonemes as P  # noqa: E402
from mimetic.data import track  # noqa: E402
from mimetic.face import anatomy as A  # noqa: E402
from mimetic.face import canonical as C  # noqa: E402
from mimetic.model.articulation import measures  # noqa: E402
from mimetic.model.audio_model import HOP, N_FFT, SR  # noqa: E402
from mimetic.model.controller import EMOTIONS  # noqa: E402
from mimetic.paths import EXPERIMENTS, PROCESSED  # noqa: E402
from mimetic.runtime.face_model import load_face_model  # noqa: E402
from mimetic.runtime.live import AudioDriver, LivePost  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import geo_features, train_emotion_classifier  # noqa: E402
from train_controller import Data, to_dev  # noqa: E402

LIPS = C.UPPER_INNER + C.LOWER_INNER + C.UPPER_OUTER + C.LOWER_OUTER
SPARED = [i for i, n in enumerate(A.MUSCLE_NAMES) if n.startswith("orbicularis_oris") or n.startswith("jaw_")]


def emotion_offsets(train: Data) -> np.ndarray:
    acts = train.arr["acts"].astype(np.float32)
    z = np.log(np.clip(acts, 1e-3, 1 - 1e-3) / np.clip(1 - acts, 1e-3, 1))
    by_spk: dict[str, dict[int, list]] = {}
    for c in train.clips:
        if c["dataset"] not in ("ravdess", "cremad"):
            continue
        e = 0 if c["emotion"] == "neutral" else (c["emo"] if c["inten"] > 0 else None)
        if e is not None:
            by_spk.setdefault(c["speaker"], {}).setdefault(e, []).append(z[c["offset"]:c["offset"] + c["n"]])
    off = np.zeros((len(EMOTIONS), A.N_ACT), np.float32)
    cnt = np.zeros(len(EMOTIONS))
    for d in by_spk.values():
        if 0 not in d:
            continue
        base = np.concatenate(d[0]).mean(0)
        for e, xs in d.items():
            if e:
                off[e] += np.concatenate(xs).mean(0) - base
                cnt[e] += 1
    off /= np.maximum(cnt, 1)[:, None]
    off[:, SPARED] = 0
    return off


def main():
    import os
    dev = os.environ.get("MIMETIC_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    train = Data("train")
    offsets = emotion_offsets(train)
    top = {EMOTIONS[e]: [A.MUSCLE_NAMES[k] for k in np.argsort(-np.abs(offsets[e]))[:3]] for e in range(1, len(EMOTIONS))}
    print("strongest emotion offsets:", top, flush=True)

    face = load_face_model(dev)
    pca = np.load(MO.MOTION / "pca.npz")
    comps, mean = torch.tensor(pca["comps"], device=dev), torch.tensor(pca["mean"], device=dev)
    drv = AudioDriver.load(dev, post=False)
    rng = np.random.default_rng(7)
    bil = [P.PID[p] for p in ("P", "B", "M")]

    def collect(idx):
        raws, gts, phones, emos = [], [], [], []
        for i in idx:
            c = train.clips[i]
            aud = track.load_clip(PROCESSED / c["dataset"] / c["speaker"] / f"{c['clip']}.npz")["audio"]
            out = drv.raw(aud, ("auto", 1.0))
            la = drv.m.lookahead
            raws.append((out["acts"][0, la:].cpu().numpy(), out["emotion"][0, la:].cpu().numpy(),
                         torch.sigmoid(out["vad"][0, la:]).cpu().numpy(), out["phone"][0, la:].cpu().numpy()))
            b = to_dev(train.collate([c], None, maxt=100000), dev)
            gts.append(face.rest + (b["coef"][0] @ comps + mean).view(b["coef"].shape[1], -1, 3))
            phones.append(b["phone"][0].cpu().numpy())
            emos.append(c["emo"] if c["dataset"] in ("ravdess", "cremad") else -1)
        return raws, gts, phones, emos

    raws, gts, phones, emos = collect(rng.choice(len(train.clips), 300, replace=False))
    clf = None

    def score(post, gain=1.0):
        ap_c, jerks, gt_jerks, lags, hit, tot, e_hit, e_tot = [], [], [], [], 0, 0, 0, 0
        for (acts, emo, vad, phl), gt, ph, lab in zip(raws, gts, phones, emos):
            a = post.fresh()(acts, emo, vad, gain, phl) if post is not None else acts
            t_in = (np.arange(len(a)) * HOP + N_FFT / 2) / SR
            t_out = np.arange(len(gt)) / 30.0
            a30 = np.stack([np.interp(t_out, t_in, a[:, k]) for k in range(A.N_ACT)], -1).astype(np.float32)
            with torch.no_grad():
                x = face(torch.tensor(a30, device=dev))
            pp, gg = measures(x)["aperture"].cpu().numpy(), measures(gt)["aperture"].cpu().numpy()
            if clf is not None and lab >= 0:
                with torch.no_grad():
                    f = geo_features(x[None], face.rest)
                    e_hit += int(clf(f, torch.ones(f.shape[:2], device=dev)).argmax(-1).item() == lab)
                e_tot += 1
            n = len(gg)
            if pp.std() > 1e-6 and gg.std() > 1e-6:
                ap_c.append(np.corrcoef(pp, gg)[0, 1])
                if n > 30:
                    ls = range(-6, 7)
                    cs = [np.corrcoef(pp[max(0, L):n + min(0, L)], gg[max(0, -L):n - max(0, L)])[0, 1] for L in ls]
                    lags.append(ls[int(np.nanargmax(cs))])
            for arr, dst in ((x[:, LIPS], jerks), (gt[:, LIPS], gt_jerks)):
                jk = arr[3:] - 3 * arr[2:-1] + 3 * arr[1:-2] - arr[:-3]
                dst.append((jk ** 2).sum(-1).mean().item())
            isb = np.isin(ph, bil)
            t = 0
            while t < n:
                if isb[t]:
                    s0 = t
                    while t < n and isb[t]:
                        t += 1
                    tot += 1
                    hit += int(pp[max(0, s0 - 1):t + 1].min() < 0.10)
                else:
                    t += 1
        return {"ApCorr": float(np.mean(ap_c)), "Jerk_rel": float(np.mean(jerks) / np.mean(gt_jerks)),
                "AV_ms": 1000 / 30 * float(np.mean(lags)), "Closure": hit / max(1, tot),
                "EmoAcc": e_hit / e_tot if e_tot else None}

    results = {"raw": score(None)}
    print("raw", results["raw"], flush=True)
    for mc in (1.0, 2.0, 3.0, 5.0, 8.0):
        for beta in (0.5, 2.0, 8.0):
            r = score(LivePost(offsets, mc, beta))
            results[f"mc{mc}_b{beta}"] = r
            print(f"min_cutoff {mc} beta {beta}", r, flush=True)
    ok = {k: r for k, r in results.items() if k != "raw" and r["Jerk_rel"] <= 1.0
          and abs(r["AV_ms"] - results["raw"]["AV_ms"]) <= 1000 / 30}
    best = max(ok, key=lambda k: ok[k]["ApCorr"]) if ok else min(results, key=lambda k: results[k]["Jerk_rel"])
    mc, beta = (float(best.split("_")[0][2:]), float(best.split("_b")[1])) if best != "raw" else (100.0, 0.0)
    print("chosen filter", best, results[best], flush=True)

    # --- emotion pathway, on emotional training clips
    emo_idx = [i for i, c in enumerate(train.clips) if c["dataset"] in ("ravdess", "cremad")]
    raws, gts, phones, emos = collect(rng.choice(emo_idx, 300, replace=False))
    pstd = torch.tensor(train.arr["coef"][::50].astype(np.float32).std(0), device=dev).clamp_min(1e-3)
    torch.manual_seed(0)
    clf = train_emotion_classifier(train, comps, mean, pstd, dev, face)
    emo_res = {"filter_only": score(LivePost(offsets, mc, beta), gain=0.0)}
    print("filter only", emo_res["filter_only"], flush=True)
    for tau in (0.3, 0.8):
        for sh in (1.0, 2.0):
            for g in (1.0, 1.5):
                r = score(LivePost(offsets, mc, beta, emo_tau=tau, sharpen=sh, gain=g))
                emo_res[f"tau{tau}_sh{sh}_g{g}"] = r
                print(f"tau {tau} sharpen {sh} gain {g}", r, flush=True)
    floor = emo_res["filter_only"]["Closure"] - 0.03
    ok = {k: r for k, r in emo_res.items() if k != "filter_only" and r["Closure"] >= floor}
    emo_best = (max(ok, key=lambda k: ok[k]["EmoAcc"]) if ok else
                max((k for k in emo_res if k != "filter_only"), key=lambda k: emo_res[k]["Closure"]))
    tau, sh, g = (float(v[len(p):]) for v, p in zip(emo_best.split("_"), ("tau", "sh", "g")))
    print("chosen emotion", emo_best, emo_res[emo_best])
    np.savez(LivePost.PATH, offsets=offsets, min_cutoff=mc, beta=beta, emo_tau=tau, sharpen=sh, gain=g)
    (EXPERIMENTS / "live_post_fit.json").write_text(json.dumps(
        {"chosen_filter": best, "filter_sweep": results, "chosen_emotion": emo_best, "emotion_sweep": emo_res,
         "emotion_top_muscles": top}, indent=1))


if __name__ == "__main__":
    main()
