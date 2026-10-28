"""Side-by-side: tracked real face (rendered on the simulator mesh) vs its muscle fit, with muscle overlay."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from mimetic.data import motion as MO  # noqa: E402
from mimetic.face import anatomy as A  # noqa: E402
from mimetic.face.render import FaceRenderer  # noqa: E402
from mimetic.face.sim import FaceSim  # noqa: E402
from mimetic.runtime.export3d import jaw_tracks  # noqa: E402
from mimetic.runtime.face_model import load_face_model  # noqa: E402


def main(out="experiments/fit_examples.png"):
    face = load_face_model("cpu")
    sim = FaceSim(calibrate=False)
    r = FaceRenderer(sim, (224, 224))
    pca = np.load(MO.MOTION / "pca.npz")
    picks = [("ravdess_02", "happy"), ("ravdess_03", "angry"), ("cremad_1015", "sad"), ("ravdess_05", "surprised"),
             ("cremad_1020", "disgusted"), ("grid_s1", "neutral")]
    rows = []
    for spk, emo in picks:
        p = MO.MOTION / f"{spk}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        index = json.loads(z["index"].tobytes().decode())
        acts = np.load(MO.MOTION / f"{spk}_acts.npy").astype(np.float32)
        cands = [c for c in index if c["emotion"] == emo and c.get("intensity", 1) >= 1] or index
        c = cands[len(cands) // 2]
        o, n = c["offset"], c["n"]
        # frame with the largest expression
        coef = z["coef"][o:o + n].astype(np.float32)
        k = int(np.argmax(np.abs(coef[:, :8]).sum(1)))
        disp = (coef[k] @ pca["comps"] + pca["mean"]).reshape(468, 3)
        gt = face.rest.numpy() + disp
        a = acts[o + k]
        x = face(torch.tensor(a[None]))[0].numpy()
        J = jaw_tracks(a[None])[0]
        jaw = (J[:3, :3], J[:3, 3])
        tiles = [r.render(gt), r.render(x, act=a, jaw=jaw), r.render(x, act=a, jaw=jaw, show_muscles=True, skin_alpha=0.6)]
        top = sorted(range(A.N_ACT), key=lambda i: -a[i])[:4]
        rows.append((f"{spk} {emo}", tiles, ", ".join(f"{A.MUSCLE_NAMES[i]} {a[i]:.2f}" for i in top)))
    W = 224
    sheet = Image.new("RGB", (3 * W, len(rows) * (W + 28)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (lab, tiles, top) in enumerate(rows):
        y = i * (W + 28)
        d.text((4, y + 2), f"{lab}  |  tracked  ->  muscle fit  ->  muscles", fill=(0, 0, 0))
        d.text((4, y + 14), top, fill=(90, 90, 90))
        for j, t in enumerate(tiles):
            sheet.paste(Image.fromarray(t), (j * W, y + 28))
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
