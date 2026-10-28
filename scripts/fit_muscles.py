"""Fit muscle activations to every tracked frame with the physics-decoder autoencoder.

  python scripts/fit_muscles.py --epochs 12 [--calibrate]

Writes data/motion/{speaker}_acts.npy (float16, aligned with the motion DB
frames), checkpoints/encoder.pt, and a fit report in experiments/fit_report.json.
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

from mimetic.data import motion as MO  # noqa: E402
from mimetic.face import anatomy as A  # noqa: E402
from mimetic.face import canonical as C  # noqa: E402
from mimetic.face.surrogate import MuscleSurrogate  # noqa: E402
from mimetic.fit.encoder import Calibration, TemporalEncoder, fit_weights  # noqa: E402
from mimetic.paths import CHECKPOINTS, EXPERIMENTS  # noqa: E402

WIN = 64


def load_db():
    sp = np.load(MO.MOTION / "speakers.npz")
    pca = np.load(MO.MOTION / "pca.npz")
    spk = []
    for name in sp["names"]:
        p = MO.MOTION / f"{name}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        index = json.loads(z["index"].tobytes().decode())
        spk.append({"name": str(name), "coef": z["coef"], "index": index, "bs": z["bs"]})
    return spk, pca


def windows(spk, rng=None):
    """(speaker_idx, start) for 64-frame windows covering every clip."""
    out = []
    for si, s in enumerate(spk):
        for c in s["index"]:
            o, n = c["offset"], c["n"]
            starts = list(range(0, max(1, n - WIN + 1), WIN // 2))
            if n > WIN and starts[-1] != n - WIN:
                starts.append(n - WIN)
            for st in starts:
                out.append((si, o + st, min(WIN, n - st if n >= WIN else n)))
    return out


def batch(spk, wins, pca_std, dev):
    B = len(wins)
    K = spk[0]["coef"].shape[1]
    c = np.zeros((B, WIN, K), np.float32)
    m = np.zeros((B, WIN), np.float32)
    for i, (si, o, n) in enumerate(wins):
        c[i, :n] = spk[si]["coef"][o:o + n]
        m[i, :n] = 1
    return torch.tensor(c / pca_std, device=dev), torch.tensor(m, device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--l1", type=float, default=0.0015)
    ap.add_argument("--aperture", type=float, default=30.0)
    ap.add_argument("--eye-aperture", type=float, default=30.0)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=48)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    spk, pca = load_db()
    print(f"{len(spk)} speakers, {sum(len(s['coef']) for s in spk)} frames", flush=True)
    comps = torch.tensor(pca["comps"], device=dev)
    mean = torch.tensor(pca["mean"], device=dev)
    # Normalise PCA coefficients by their spread so every mode is equally visible to the encoder.
    pca_std = np.std(np.concatenate([s["coef"][:2000] for s in spk]).astype(np.float32), 0)
    pca_std = np.maximum(pca_std, 1e-4).astype(np.float32)

    ck = torch.load(CHECKPOINTS / "surrogate.pt", map_location="cpu")
    calib = torch.load(CHECKPOINTS / "sim_calibration.pt", map_location="cpu")
    from mimetic.face.sim import FaceSim
    sim = FaceSim(calibrate=False)
    sim.F = calib["F"]
    sur = MuscleSurrogate.from_sim(sim)
    sur.load_state_dict(ck["state"])
    sur = sur.to(dev).eval()
    for p in sur.parameters():
        p.requires_grad_(False)
    rest = sur.rest
    W = fit_weights(sim.X.numpy()).to(dev)
    # Upper-face muscles make small movements that matter for emotion: a lighter effort prior there.
    upper = ("frontalis", "corrugator", "depressor_supercilii", "procerus", "orbicularis_oculi", "levator_palpebrae")
    l1w = torch.tensor([0.3 if n.startswith(upper) else 1.0 for n in A.MUSCLE_NAMES], device=dev)
    enc = TemporalEncoder(k_in=comps.shape[0]).to(dev)
    cal = Calibration(sim.X.numpy(), sim.W.numpy(), sim.w_jaw.numpy(), sim.face).to(dev) if args.calibrate else None
    params = list(enc.parameters()) + (list(cal.parameters()) if cal is not None else [])
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    wins = windows(spk)
    rng = np.random.default_rng(0)
    rng.shuffle(wins)
    n_val = min(1500, len(wins) // 20)
    val, train = wins[:n_val], wins[n_val:]
    steps = args.epochs * (len(train) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    lips = C.UPPER_INNER + C.LOWER_INNER + C.UPPER_OUTER + C.LOWER_OUTER

    def decode(a):
        x = sur(a)
        if cal is not None:
            x = x + cal(a)
        return x

    def step_loss(cb, mb, train=True):
        a = enc(cb)
        x = decode(a)                                                   # (B,T,N,3)
        disp = (cb * torch.tensor(pca_std, device=dev)) @ comps + mean
        y = rest + disp.view(*cb.shape[:2], -1, 3)
        err = ((x - y) ** 2 * W).sum(-1).mean(-1)                         # (B,T)
        rec = (err * mb).sum() / mb.sum()
        l1 = ((a * l1w).sum(-1) * mb).sum() / mb.sum()
        da = a[:, 1:] - a[:, :-1]
        sm = ((da ** 2).sum(-1) * mb[:, 1:] * mb[:, :-1]).sum() / mb.sum()
        # Lip contact matters more than its share of vertices: match each inner-lip gap explicitly,
        # and never let the lips pass through each other.
        ap_x, ap_y = MO.pair_apertures_t(x), MO.pair_apertures_t(y)
        l_ap = ((((ap_x - ap_y) ** 2).mean(-1) + 4 * torch.relu(-ap_x).pow(2).mean(-1)) * mb).sum() / mb.sum()
        # Eye openness is a 1-2 mm cue that carries surprise, fear and the Duchenne squint.
        l_eye = (((MO.lid_apertures_t(x) - MO.lid_apertures_t(y)) ** 2).mean(-1) * mb).sum() / mb.sum()
        loss = rec + args.l1 * l1 + args.smooth * sm + args.aperture * l_ap + args.eye_aperture * l_eye
        if cal is not None:
            loss = loss + 0.5 * cal.penalty()
        with torch.no_grad():
            ve = (x - y).norm(dim=-1)
            lip_mm = ((ve[..., lips].mean(-1) * mb).sum() / mb.sum()).item() * 10
            all_mm = ((ve.mean(-1) * mb).sum() / mb.sum()).item() * 10
        return loss, rec.item(), lip_mm, all_mm, a

    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        enc.train()
        rng.shuffle(train)
        for j in range(0, len(train) - args.bs + 1, args.bs):
            cb, mb = batch(spk, train[j:j + args.bs], pca_std, dev)
            loss, *_ = step_loss(cb, mb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); step += 1
        enc.eval()
        with torch.no_grad():
            stats = []
            for j in range(0, len(val), 64):
                cb, mb = batch(spk, val[j:j + 64], pca_std, dev)
                _, rec, lip_mm, all_mm, a = step_loss(cb, mb, False)
                stats.append((rec, lip_mm, all_mm, (a * mb[..., None]).sum().item() / mb.sum().item()))
            s = np.mean(stats, 0)
        print(f"epoch {ep + 1}: val rec {s[0]:.4f}  lips {s[1]:.3f} mm  all {s[2]:.3f} mm  "
              f"mean act sum {s[3]:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    torch.save({"enc": enc.state_dict(), "pca_std": pca_std, "cal": cal.state_dict() if cal is not None else None,
                "k_in": comps.shape[0]}, CHECKPOINTS / "encoder.pt")
    # Infer every frame (full clips, no windowing artefacts: run per clip).
    report = {"speakers": {}, "args": vars(args)}
    enc.eval()
    with torch.no_grad():
        for s in spk:
            acts = np.zeros((len(s["coef"]), A.N_ACT), np.float16)
            errs = []
            for c in s["index"]:
                o, n = c["offset"], c["n"]
                cb = torch.tensor(s["coef"][o:o + n][None].astype(np.float32) / pca_std, device=dev)
                a = enc(cb)
                acts[o:o + n] = a[0].cpu().numpy().astype(np.float16)
                x = decode(a)[0]
                disp = (cb[0] * torch.tensor(pca_std, device=dev)) @ comps + mean
                y = rest + disp.view(n, -1, 3)
                ve = (x - y).norm(dim=-1)
                errs.append([ve[:, lips].mean().item() * 10, ve.mean().item() * 10])
            np.save(MO.MOTION / f"{s['name']}_acts.npy", acts)
            e = np.mean(errs, 0)
            report["speakers"][s["name"]] = {"lips_mm": float(e[0]), "all_mm": float(e[1])}
    vals = np.array([[v["lips_mm"], v["all_mm"]] for v in report["speakers"].values()])
    report["mean_lips_mm"], report["mean_all_mm"] = float(vals[:, 0].mean()), float(vals[:, 1].mean())
    EXPERIMENTS.mkdir(exist_ok=True)
    (EXPERIMENTS / "fit_report.json").write_text(json.dumps(report, indent=1))
    print(f"done: mean residual lips {report['mean_lips_mm']:.3f} mm, all {report['mean_all_mm']:.3f} mm", flush=True)


if __name__ == "__main__":
    main()
