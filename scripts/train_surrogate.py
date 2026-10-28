"""Generate simulator samples and train the neural muscle surrogate.

  python scripts/train_surrogate.py --samples 160000 [--real data/motion/fitted_acts.npy]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mimetic.face import anatomy as A  # noqa: E402
from mimetic.face import canonical as C  # noqa: E402
from mimetic.face.sim import FaceSim  # noqa: E402
from mimetic.face.surrogate import MuscleSurrogate, sample_activations  # noqa: E402
from mimetic.paths import CHECKPOINTS, DATA  # noqa: E402


def region_weights(X):
    w = np.ones(len(X))
    w[C.UPPER_INNER + C.LOWER_INNER + C.UPPER_OUTER + C.LOWER_OUTER] = 4.0
    w[C.R_UPPER_LID + C.R_LOWER_LID + C.L_UPPER_LID + C.L_LOWER_LID] = 2.0
    return torch.tensor(w, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=160000)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--real", default=None, help="fitted real activations (.npy) to mix into sampling")
    ap.add_argument("--out", default=str(CHECKPOINTS / "surrogate.pt"))
    ap.add_argument("--cache", default=str(DATA / "sim" / "surrogate_data.pt"))
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    sim = FaceSim(device=dev)
    torch.save({"F": sim.F.cpu(), "rest": sim.X.cpu()}, CHECKPOINTS / "sim_calibration.pt")
    print(f"sim calibrated (max excursion error {sim.calib_error:.2f})", flush=True)
    real = torch.tensor(np.load(args.real), dtype=torch.float32) if args.real else None

    if Path(args.cache).exists() and real is None:
        blob = torch.load(args.cache)
        acts, soft = blob["acts"], blob["soft"]
        print(f"loaded {len(acts)} cached samples", flush=True)
    else:
        g = torch.Generator().manual_seed(1234)
        acts, soft = [], []
        t0 = time.time()
        B = 2048
        sur0 = MuscleSurrogate.from_sim(sim)
        for i in range(0, args.samples, B):
            a = sample_activations(B, g, real)
            x = sim.solve(a.to(dev), iters=args.iters).cpu()
            T = sur0.skinned(a[:, -5:])
            acts.append(a)
            soft.append((x - T).half())
            if (i // B) % 10 == 0:
                print(f"  sim {i + B}/{args.samples} ({time.time() - t0:.0f}s)", flush=True)
        acts, soft = torch.cat(acts), torch.cat(soft)
        torch.save({"acts": acts, "soft": soft}, args.cache)

    n_val = 8000
    acts_v, soft_v = acts[:n_val].to(dev), soft[:n_val].float().to(dev)
    acts_t, soft_t = acts[n_val:], soft[n_val:]
    model = MuscleSurrogate.from_sim(sim).to(dev)
    # Initialise the linear basis by least squares (fast, good starting point).
    X = acts_t[:40000].to(dev)
    Y = soft_t[:40000].float().reshape(len(X), -1).to(dev)
    sol = torch.linalg.lstsq(X.cpu(), Y.cpu()).solution
    model.lin.weight.data = sol.T.to(dev)
    w = region_weights(sim.X.cpu().numpy()).to(dev)[None, :, None]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    steps = args.epochs * (len(acts_t) // 1024)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(len(acts_t))
        model.train()
        for j in range(0, len(perm) - 1023, 1024):
            idx = perm[j:j + 1024]
            a = acts_t[idx].to(dev)
            y = soft_t[idx].float().to(dev)
            pred = model.soft(a)
            loss = ((pred - y) ** 2 * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); step += 1
        model.eval()
        with torch.no_grad():
            e = (model.soft(acts_v) - soft_v).norm(dim=-1)
            lips = C.UPPER_INNER + C.LOWER_INNER + C.UPPER_OUTER + C.LOWER_OUTER
            print(f"epoch {ep + 1}: val mean {e.mean() * 10:.3f} mm, p99 {e.flatten().quantile(0.99) * 10:.3f} mm, "
                  f"lips {e[:, lips].mean() * 10:.3f} mm ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": model.state_dict(), "n_act": A.N_ACT, "names": A.MUSCLE_NAMES}, args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
