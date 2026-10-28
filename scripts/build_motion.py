"""Build the motion database from tracked clips (see mimetic/data/motion.py for the layout).

  python scripts/build_motion.py            # both passes over every tracked speaker
  python scripts/build_motion.py --pass a   # neutrals + PCA only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from mimetic.data import motion as MO  # noqa: E402
from mimetic.paths import ASSETS  # noqa: E402


def pass_a_speaker(sdir):
    sdir = Path(sdir)
    clips = MO.load_speaker(sdir)
    if not clips:
        return None
    N, n_neu = MO.speaker_neutral(clips)
    rest = np.load(ASSETS / "rest_neutral.npy")
    rest_gap = MO.pair_apertures(rest[None])[0]
    disps = [MO.clip_displacements(d, N)[0] for d in clips]
    closed, neutral = MO.lip_contact_params([N[None] + D for D in disps], N)
    rng = np.random.default_rng(abs(hash(sdir.name)) % 2**32)
    samples = []
    for D in disps:
        D = MO.apply_lip_contact(D, N, closed, neutral, rest_gap)
        D = MO.apply_lid_transfer(D, N, rest)
        k = max(1, len(D) // 20)
        samples.append(D[rng.choice(len(D), size=min(k, len(D)), replace=False)])
    S = np.concatenate(samples)
    if len(S) > 1500:
        S = S[rng.choice(len(S), 1500, replace=False)]
    return (sdir.name, str(sdir.parent.name), N.astype(np.float32), n_neu, S.astype(np.float32), len(clips),
            closed.astype(np.float32), neutral.astype(np.float32))


def pass_b_speaker(args):
    sdir, N, mean, comps, closed, neutral = args
    rest = np.load(ASSETS / "rest_neutral.npy")
    rest_gap = MO.pair_apertures(rest[None])[0]
    sdir = Path(sdir)
    clips = MO.load_speaker(sdir)
    rec = {k: [] for k in ("coef", "rot", "trans", "scale", "iris", "bs")}
    index = []
    off = 0
    err = []
    for d in clips:
        D, (s, R, t), iris = MO.clip_displacements(d, N)
        D = MO.apply_lip_contact(D, N, closed, neutral, rest_gap)
        D = MO.apply_lid_transfer(D, N, rest)
        flat = D.reshape(len(D), -1)
        c = (flat - mean) @ comps.T
        recon = c @ comps + mean
        err.append(np.linalg.norm((recon - flat).reshape(len(D), 468, 3), axis=-1).mean())
        rec["coef"].append(c.astype(np.float16))
        rec["rot"].append(MO.rotvec(R).astype(np.float32))
        rec["trans"].append(t.astype(np.float32))
        rec["scale"].append(s.astype(np.float32))
        rec["iris"].append((iris - N[None, :1] * 0).astype(np.float16))
        rec["bs"].append(d["bs"].astype(np.float16))
        index.append({"path": d["path"], "clip": str(d["clip"]), "offset": off, "n": len(D),
                      "fps": float(d["fps"]), "emotion": str(d["emotion"]), "intensity": int(d["intensity"]),
                      "text": str(d["text"]), "dataset": str(d["dataset"])})
        off += len(D)
    out = {k: np.concatenate(v) for k, v in rec.items()}
    out["neutral"] = N
    out["index"] = np.frombuffer(json.dumps(index).encode(), np.uint8)
    MO.MOTION.mkdir(parents=True, exist_ok=True)
    np.savez(MO.MOTION / f"{sdir.name}.npz", **out)
    return sdir.name, off, float(np.mean(err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", default="ab")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    MO.MOTION.mkdir(parents=True, exist_ok=True)
    dirs = [str(p) for p in MO.speaker_dirs()]
    t0 = time.time()
    if "a" in args.which:
        with ProcessPoolExecutor(args.workers) as ex:
            res = [r for r in ex.map(pass_a_speaker, dirs) if r is not None]
        names = [r[0] for r in res]
        neutrals = np.stack([r[2] for r in res])
        S = np.concatenate([r[4] for r in res]).reshape(-1, 468 * 3)
        mean = S.mean(0)
        U, sv, Vt = np.linalg.svd(S - mean, full_matrices=False)
        var = sv ** 2 / (sv ** 2).sum()
        comps = Vt[:MO.K_PCA]
        np.savez(MO.MOTION / "speakers.npz", names=np.array(names), datasets=np.array([r[1] for r in res]),
                 neutrals=neutrals, mean_neutral=neutrals.mean(0), n_neutral_frames=np.array([r[3] for r in res]),
                 n_clips=np.array([r[5] for r in res]), lip_closed=np.stack([r[6] for r in res]),
                 lip_neutral=np.stack([r[7] for r in res]))
        np.savez(MO.MOTION / "pca.npz", mean=mean.astype(np.float32), comps=comps.astype(np.float32),
                 var=var[:MO.K_PCA].astype(np.float32))
        print(f"pass A: {len(names)} speakers, {len(S)} samples, PCA {MO.K_PCA} explains "
              f"{var[:MO.K_PCA].sum() * 100:.2f}% ({time.time() - t0:.0f}s)", flush=True)
    if "b" in args.which:
        sp = np.load(MO.MOTION / "speakers.npz")
        pca = np.load(MO.MOTION / "pca.npz")
        nm = {str(n): (N, c, u) for n, N, c, u in zip(sp["names"], sp["neutrals"], sp["lip_closed"], sp["lip_neutral"])}
        jobs = [(d, nm[Path(d).name][0], pca["mean"], pca["comps"], nm[Path(d).name][1], nm[Path(d).name][2])
                for d in dirs if Path(d).name in nm]
        tot = 0
        with ProcessPoolExecutor(args.workers) as ex:
            for name, n, e in ex.map(pass_b_speaker, jobs):
                tot += n
                print(f"  {name}: {n} frames, PCA recon err {e * 10:.3f} mm", flush=True)
        print(f"pass B: {tot} frames ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
