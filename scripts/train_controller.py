"""Train the speech + emotion muscle controller.

  python scripts/train_controller.py --steps 30000 [--no-artic] [--single-trunk] [--name ctrl]

Losses: muscle-activation imitation, skin-vertex error through the physics
surrogate, articulation constraints (lip seal, labiodental contact, rounding,
jaw opening), jerk, and head motion.
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

from mimetic.face import anatomy as A  # noqa: E402
from mimetic.face import canonical as C  # noqa: E402
from mimetic.model.articulation import ArticulationLoss  # noqa: E402
from mimetic.model.controller import EMOTIONS, SpeechEmotionController  # noqa: E402
from mimetic.paths import CHECKPOINTS, DATA, EXPERIMENTS  # noqa: E402
from mimetic.runtime.face_model import load_face_model  # noqa: E402

TEST_SPEAKERS = {"ravdess_23", "ravdess_24", "grid_s33", "grid_s34", "grid_s32"} | \
                {f"cremad_{i}" for i in (1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091)}
MAXT = 150


def intensity_of(m):
    if m["emotion"] == "neutral":
        return 0.0
    if m["dataset"] == "ravdess":
        return {1: 0.6, 2: 1.0}.get(m["intensity"], 0.6)
    if m["dataset"] == "cremad":
        return {1: 0.4, 2: 0.7, 3: 1.0}.get(m["intensity"], 0.7)
    return 0.6


class Data:
    def __init__(self, split):
        z = np.load(DATA / "ctrl" / "dataset.npz")
        meta = json.loads((DATA / "ctrl" / "clips.json").read_text())
        self.arr = {k: z[k] for k in z.files}
        self.speakers = meta["speakers"]
        spk_id = {s: i for i, s in enumerate(self.speakers)}
        clips = [c for c in meta["clips"] if (c["speaker"] in TEST_SPEAKERS) == (split == "test")]
        self.clips = [c for c in clips if c["n"] >= 20]
        for c in self.clips:
            c["spk"] = spk_id[c["speaker"]]
            c["emo"] = EMOTIONS.index(c["emotion"]) if c["emotion"] in EMOTIONS else 0
            c["inten"] = intensity_of(c)
        by = {}
        for i, c in enumerate(self.clips):
            by.setdefault(c["dataset"], []).append(i)
        self.by_dataset = by

    def sample(self, B, rng, balanced=True):
        if balanced:
            ds = list(self.by_dataset)
            idx = [rng.choice(self.by_dataset[ds[rng.integers(len(ds))]]) for _ in range(B)]
        else:
            idx = rng.integers(0, len(self.clips), B)
        return self.collate([self.clips[i] for i in idx], rng)

    def collate(self, clips, rng=None, maxt=MAXT):
        B = len(clips)
        T = min(maxt, max(c["n"] for c in clips))
        out = {"acts": np.zeros((B, T, A.N_ACT), np.float32), "rot": np.zeros((B, T, 3), np.float32),
               "blink": np.zeros((B, T), bool), "phone": np.zeros((B, T), np.int64),
               "ppos": np.zeros((B, T), np.float32), "pdur": np.zeros((B, T), np.float32),
               "prosody": np.zeros((B, T, 4), np.float32), "mask": np.zeros((B, T), np.float32)}
        for k in ("coef", "fitcoef"):
            if k in self.arr:
                out[k] = np.zeros((B, T, self.arr[k].shape[1]), np.float32)
        for i, c in enumerate(clips):
            n = c["n"]
            st = 0 if (rng is None or n <= T) else int(rng.integers(0, n - T + 1))
            L = min(n, T)
            o = c["offset"] + st
            for k in out:
                if k != "mask" and k in self.arr:
                    out[k][i, :L] = self.arr[k][o:o + L]
            out["mask"][i, :L] = 1
        out["emo"] = np.array([c["emo"] for c in clips])
        out["inten"] = np.array([c["inten"] for c in clips], np.float32)
        out["spk"] = np.array([c["spk"] for c in clips])
        return out


def to_dev(b, dev):
    return {k: torch.as_tensor(v, device=dev) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--no-artic", action="store_true")
    ap.add_argument("--baseline", choices=["none", "direct"], default="none",
                    help="direct: same network regresses displacement PCA instead of muscles")
    ap.add_argument("--name", default="controller")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    train, test = Data("train"), Data("test")
    print(f"train {len(train.clips)} clips, test {len(test.clips)} clips", flush=True)
    face = load_face_model(dev)
    direct = args.baseline == "direct"
    from mimetic.data import motion as MO
    _pca = np.load(MO.MOTION / "pca.npz")
    fcomps, fmean = torch.tensor(_pca["comps"], device=dev), torch.tensor(_pca["mean"], device=dev)
    if direct:
        from mimetic.data import motion as MO
        pca = np.load(MO.MOTION / "pca.npz")
        comps = torch.tensor(pca["comps"], device=dev)
        pmean = torch.tensor(pca["mean"], device=dev)
        pstd = torch.tensor(train.arr["coef"][::50].astype(np.float32).std(0), device=dev).clamp_min(1e-3)
        model = SpeechEmotionController(n_speakers=len(train.speakers), n_act=comps.shape[0], output="linear").to(dev)
    else:
        model = SpeechEmotionController(n_speakers=len(train.speakers)).to(dev)
    n_avg = len(train.speakers)
    rest = face.rest
    art = ArticulationLoss(rest, A.MUSCLE_NAMES.index("jaw_open"))
    lips = C.UPPER_INNER + C.LOWER_INNER + C.UPPER_OUTER + C.LOWER_OUTER
    vw = torch.ones(rest.shape[0], device=dev)
    vw[lips] = 4.0
    vw[C.R_UPPER_LID + C.R_LOWER_LID + C.L_UPPER_LID + C.L_LOWER_LID] = 1.5
    pal = [A.MUSCLE_NAMES.index(n) for n in ("orbicularis_oculi_palpebral_R", "orbicularis_oculi_palpebral_L")]
    jaw_i = A.MUSCLE_NAMES.index("jaw_open")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)

    def decode_direct(c):
        return rest + ((c * pstd) @ comps + pmean).view(*c.shape[:-1], -1, 3)

    log = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        b = to_dev(train.sample(args.bs, rng), dev)
        spk = torch.where(torch.rand(args.bs, device=dev) < 0.3, torch.full_like(b["spk"], n_avg), b["spk"])
        keep = (torch.rand(args.bs, device=dev) < 0.5).float()
        ctx = (torch.rand(args.bs, device=dev) < 0.7).float()
        out, head = model(b["phone"], b["ppos"], b["pdur"], b["prosody"], b["emo"], b["inten"], spk,
                          pad_mask=b["mask"] == 0, emo_prosody_keep=keep, emo_ctx_keep=ctx)
        m = b["mask"]
        if direct:
            tgt = b["coef"] / pstd
            l_act = (((out - tgt) ** 2).mean(-1) * m).sum() / m.sum()
            x_all = decode_direct(out)
            x_t = decode_direct(tgt)
            l_vert = ((((x_all - x_t) ** 2).sum(-1) * vw).mean(-1) * m).sum() / m.sum()
            jaw_proxy = torch.relu(rest[14, 1] - x_all[..., 14, 1]) / 2.0     # lower-lip drop as a jaw proxy
            acts_like = torch.zeros(*out.shape[:2], A.N_ACT, device=dev)
            acts_like[..., jaw_i] = jaw_proxy
            seq = out
        else:
            tgt = b["acts"]
            w_act = torch.ones_like(tgt)
            w_act[..., pal] = torch.where(b["blink"][..., None], 0.0, 1.0)   # blinks are added stochastically at runtime
            l_act = ((torch.nn.functional.smooth_l1_loss(out, tgt, beta=0.05, reduction="none") * w_act).mean(-1) * m).sum() / m.sum()
            x_all = face(out.reshape(-1, A.N_ACT)).view(*out.shape[:2], -1, 3)
            x_t = rest + (b["fitcoef"] @ fcomps + fmean).view(*tgt.shape[:2], -1, 3)   # precomputed fit faces
            l_vert = ((((x_all - x_t) ** 2).sum(-1) * vw).mean(-1) * m).sum() / m.sum()
            acts_like = out
            seq = out
        if args.no_artic:
            l_art, st = torch.zeros((), device=dev), {"closure_rate": 0.0}
        else:
            l_art, st = art(x_all, acts_like, b["phone"], b["ppos"], m)
        jerk = seq[:, 3:] - 3 * seq[:, 2:-1] + 3 * seq[:, 1:-2] - seq[:, :-3]
        l_jerk = ((jerk ** 2).sum(-1) * m[:, 3:]).sum() / m.sum()
        l_head = ((head - b["rot"]).abs().sum(-1) * m).sum() / m.sum()
        loss = l_act + 2.0 * l_vert + l_art + (0.05 if direct else 0.5) * l_jerk + 0.3 * l_head
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 250 == 0:
            rec = {"step": step, "loss": loss.item(), "act": l_act.item(), "vert": l_vert.item(), "art": l_art.item(),
                   "jerk": l_jerk.item(), "head": l_head.item(), "closure": st["closure_rate"], "t": time.time() - t0}
            log.append(rec)
            print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
        if step % 5000 == 0 or step == args.steps:
            ck = {"model": model.state_dict(), "n_speakers": len(train.speakers), "speakers": train.speakers,
                  "args": vars(args)}
            if direct:
                ck["pstd"] = pstd.cpu()
            torch.save(ck, CHECKPOINTS / f"{args.name}.pt")
    (EXPERIMENTS / f"{args.name}_trainlog.json").write_text(json.dumps(log))
    print("saved", CHECKPOINTS / f"{args.name}.pt", flush=True)


if __name__ == "__main__":
    main()
