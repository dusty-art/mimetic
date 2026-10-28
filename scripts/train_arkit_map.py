"""Learn muscles -> ARKit blendshapes from frames where both a muscle fit and tracked blendshapes exist.

Gaze channels (eyeLook*) are excluded: they are driven by the gaze track, not by muscles.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mimetic.data import motion as MO  # noqa: E402
from mimetic.paths import CHECKPOINTS, EXPERIMENTS  # noqa: E402
from mimetic.runtime.outputs import ARKIT, ArkitMap  # noqa: E402

TEST = ("ravdess_23", "ravdess_24", "grid_s33", "grid_s34", "grid_s32", "cremad_1090", "cremad_1091")


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sp = np.load(MO.MOTION / "speakers.npz")
    tr_a, tr_b, te_a, te_b = [], [], [], []
    for n in sp["names"]:
        p = MO.MOTION / f"{n}_acts.npy"
        if not p.exists():
            continue
        a = np.load(p).astype(np.float32)
        b = np.load(MO.MOTION / f"{n}.npz")["bs"][:, 1:].astype(np.float32)
        if str(n) in TEST:
            te_a.append(a); te_b.append(b)
        else:
            idx = np.random.default_rng(0).choice(len(a), min(len(a), 30000), replace=False)
            tr_a.append(a[idx]); tr_b.append(b[idx])
    A_, B_ = torch.tensor(np.concatenate(tr_a)), torch.tensor(np.concatenate(tr_b))
    At, Bt = torch.tensor(np.concatenate(te_a)).to(dev), torch.tensor(np.concatenate(te_b)).to(dev)
    mask = torch.tensor([0.0 if n.startswith("eyeLook") else 1.0 for n in ARKIT], device=dev)
    m = ArkitMap().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    steps = 6000
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, len(A_), (2048,))
        a, b = A_[idx].to(dev), B_[idx].to(dev)
        loss = (((m(a) - b) ** 2) * mask).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad():
        p = m(At)
        ss_res = ((p - Bt) ** 2).sum(0)
        ss_tot = ((Bt - Bt.mean(0)) ** 2).sum(0)
        r2 = (1 - ss_res / ss_tot.clamp_min(1e-6)).cpu().numpy()
    rep = {n: float(r) for n, r in zip(ARKIT, r2) if not n.startswith("eyeLook")}
    torch.save({"state": m.state_dict()}, CHECKPOINTS / "arkit_map.pt")
    EXPERIMENTS.mkdir(exist_ok=True)
    (EXPERIMENTS / "arkit_map_r2.json").write_text(json.dumps(rep, indent=1))
    good = {k: round(v, 2) for k, v in sorted(rep.items(), key=lambda kv: -kv[1])}
    print(f"held-out R^2 (median {np.median(list(rep.values())):.2f}) in {time.time() - t0:.0f}s:", good)


if __name__ == "__main__":
    main()
