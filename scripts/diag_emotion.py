"""Is emotion information preserved by the muscle fit? Cross-train emotion classifiers on GT vs fitted faces."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch, torch.nn as nn
from mimetic.data import motion as MO
from mimetic.face import anatomy as A
from mimetic.runtime.face_model import load_face_model
from train_controller import Data, to_dev
from evaluate import EmoClassifier, coef_of

dev = "mps"
train, test = Data("train"), Data("test")
face = load_face_model(dev); rest = face.rest
pca = np.load(MO.MOTION / "pca.npz")
comps, mean = torch.tensor(pca["comps"], device=dev), torch.tensor(pca["mean"], device=dev)
pstd = torch.tensor(train.arr["coef"][::50].astype(np.float32).std(0), device=dev).clamp_min(1e-3)

def feats(b, kind):
    m = b["mask"]
    if kind == "gt":
        return b["coef"] / pstd
    if kind == "fit":
        x = face(b["acts"].reshape(-1, A.N_ACT)).view(*m.shape, -1, 3)
        return coef_of(x, rest, comps, mean, pstd)
    if kind == "acts":
        return b["acts"]

def run(kind, steps=2500):
    rng = np.random.default_rng(1)
    k_in = A.N_ACT if kind == "acts" else comps.shape[0]
    clf = EmoClassifier(k_in).to(dev); opt = torch.optim.AdamW(clf.parameters(), 1e-3, weight_decay=1e-3)
    emo_clips = [i for i, c in enumerate(train.clips) if c["dataset"] in ("ravdess", "cremad")]
    for s in range(steps):
        b = to_dev(train.collate([train.clips[i] for i in rng.choice(emo_clips, 64)], rng), dev)
        with torch.no_grad(): f = feats(b, kind)
        loss = nn.functional.cross_entropy(clf(f + 0.05 * torch.randn_like(f), b["mask"]), b["emo"])
        opt.zero_grad(); loss.backward(); opt.step()
    return clf.eval()

tc = [c for c in test.clips if c["dataset"] in ("ravdess", "cremad")]
clfs = {k: run(k) for k in ("gt", "fit", "acts")}
res = {}
with torch.no_grad():
    for tk in ("gt", "fit", "acts"):
        for ck, clf in clfs.items():
            if (tk == "acts") != (ck == "acts"): continue
            hit = tot = 0
            for j in range(0, len(tc), 64):
                b = to_dev(test.collate(tc[j:j + 64], None), dev)
                hit += (clf(feats(b, tk), b["mask"]).argmax(-1) == b["emo"]).sum().item(); tot += len(tc[j:j + 64])
            res[f"train {ck} -> test {tk}"] = round(hit / tot, 3)
print(res, "chance ~", round(max(np.bincount([c["emo"] for c in tc])) / len(tc), 3))
