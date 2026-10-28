"""Report figures: RL curves, evaluation bars, emotion grid, speech/emotion composition."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from mimetic.paths import EXPERIMENTS  # noqa: E402

FIG = EXPERIMENTS / "figures"


def rl_curves():
    a = json.loads((EXPERIMENTS / "rl_articulation.json").read_text())
    long = EXPERIMENTS / "rl_ppo_long.json"
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    for key, lab, col in (("apg", "Analytic policy gradient (through the muscle face)", "#c2410c"),
                          ("ppo", "PPO", "#2563eb")):
        c = a[key]["curve"]
        ax.plot([p["env_steps"] / 1e6 for p in c], [p["goal_err"] for p in c], color=col, label=lab, lw=1.6)
    if long.exists():
        b = json.loads(long.read_text())
        c = b["ppo"]["curve"]
        ax.plot([p["env_steps"] / 1e6 for p in c], [p["goal_err"] for p in c], color="#2563eb", ls="--", lw=1.2,
                label="PPO, longer run")
    ax.set_yscale("log")
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("weighted target error (training)")
    ax.set_title("Learning to articulate with 52 muscle actuators")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "rl_curves.png")


def eval_bars():
    r = json.loads((EXPERIMENTS / "eval_results.json").read_text())
    systems = [k for k in r if not k.startswith("_")]
    labels = {"ground_truth": "real (tracked)", "muscle_fit": "muscle fit", "viseme_table": "viseme table",
              "mimetic": "Mimetic", "direct": "direct regression", "noartic": "Mimetic − constraints",
              "live_audio": "Mimetic live audio", "live_audio_raw": "live audio, unprocessed"}
    metrics = [("LVE_mm", "lip vertex error (mm) ↓"), ("Closure", "p/b/m lip closure ↑"),
               ("ApCorr", "lip aperture correlation ↑"), ("EmoAcc", "emotion recognisability ↑")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), dpi=150)
    cols = {"ground_truth": "#9ca3af", "muscle_fit": "#d1d5db", "viseme_table": "#60a5fa", "direct": "#2563eb",
            "noartic": "#fdba74", "mimetic": "#c2410c", "live_audio": "#ea580c", "live_audio_raw": "#fed7aa"}
    for ax, (m, t) in zip(axes, metrics):
        vals = [r[s][m] for s in systems]
        ax.barh([labels.get(s, s) for s in systems], vals, color=[cols.get(s, "#888") for s in systems])
        ax.set_title(t, fontsize=9)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "eval_bars.png")


def frame_at(anim, t):
    return min(anim.n - 1, int(t * anim.fps))


def renders():
    from mimetic.face.render import FaceRenderer
    from mimetic.face.sim import FaceSim
    from mimetic.runtime.engine import Engine
    from mimetic.runtime.export3d import jaw_tracks
    eng = Engine.load()
    r = FaceRenderer(FaceSim(calibrate=False), (220, 220))

    def shot(anim, k, muscles=False):
        J = jaw_tracks(anim.acts[k:k + 1])[0]
        return r.render(anim.vertices[k], act=anim.acts[k], jaw=(J[:3, :3], J[:3, 3]), show_muscles=muscles,
                        skin_alpha=0.65 if muscles else 1.0)

    def mid_of(anim, phones):
        for a, b, p in anim.segments:
            if p in phones:
                return frame_at(anim, (a + b) / 2)
        return anim.n // 2

    # Emotion grid: same sentence, one frame at the stressed vowel
    emos = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "contempt"]
    tiles = []
    for e in emos:
        tag = "" if e == "neutral" else f"[{e}:0.9] "
        anim = eng.animate_text(tag + "I can't believe it.", fps=30, seed=1)
        tiles.append((e, shot(anim, mid_of(anim, {"IY"}))))
    sheet = Image.new("RGB", (4 * 220, 2 * 236), "white")
    d = ImageDraw.Draw(sheet)
    for i, (e, t) in enumerate(tiles):
        x, y = (i % 4) * 220, (i // 4) * 236
        sheet.paste(Image.fromarray(t), (x, y + 16))
        d.text((x + 4, y + 2), e, fill=(0, 0, 0))
    sheet.save(FIG / "emotion_grid.png")

    # Speech/emotion composition: "oo" and "m" under neutral vs happy vs angry
    rows = []
    for word, phones in (("Who knew?", {"UW"}), ("Mom made more.", {"M"})):
        row = []
        for e in ("neutral", "happy", "angry"):
            tag = "" if e == "neutral" else f"[{e}:1.0] "
            anim = eng.animate_text(tag + word, fps=30, seed=2)
            k = mid_of(anim, phones)
            row.append((f"{e}: '{word}' at /{next(iter(phones))}/", shot(anim, k), shot(anim, k, True)))
        rows.append(row)
    W = 220
    sheet = Image.new("RGB", (6 * W, len(rows) * (W + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        for j, (lab, a, b) in enumerate(row):
            x, y = j * 2 * W, i * (W + 18)
            sheet.paste(Image.fromarray(a), (x, y + 18))
            sheet.paste(Image.fromarray(b), (x + W, y + 18))
            d.text((x + 4, y + 3), lab, fill=(0, 0, 0))
    sheet.save(FIG / "speech_emotion_composition.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["rl", "eval", "renders"]
    if "rl" in which:
        rl_curves()
    if "eval" in which:
        eval_bars()
    if "renders" in which:
        renders()
    print("figures in", FIG)
