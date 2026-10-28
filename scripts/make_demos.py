"""Showcase videos (experiments/demos/)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mimetic.runtime.engine import Engine
from mimetic.runtime.video import render_video

OUT = Path("experiments/demos")
OUT.mkdir(parents=True, exist_ok=True)
eng = Engine.load()


def emo_label(anim):
    def f(t):
        for a, b, e, i in anim.emotion_spans:
            if a <= t < b:
                return f"{e} {i:.1f}" if i else e
        return anim.emotion_spans[-1][2] if anim.emotion_spans else "neutral"
    return f


demos = {
    "hello": ("[happy:0.7] Hi! I'm Mimetic. Every movement you see is a muscle pulling on skin.", {}),
    "emotions": ("I can't believe you did that. [happy:0.9] I can't believe you did that. [sad:0.9] I can't believe "
                 "you did that. [angry:0.9] I can't believe you did that. [surprised:0.9] I can't believe you did that.", {}),
    "muscles": ("[happy:0.6] Watch the muscles. [disgusted:0.8] Ugh, that smells awful. [fearful:0.8] What was that noise?",
                {"show_muscles": True}),
    "bilabials": ("Peter Piper picked a peck of pickled peppers. Bob bought a big blue boat.", {}),
    "side_view": ("[happy:0.5] Who knew you would choose blue shoes? Moving smoothly through the room.", {"yaw": 55.0}),
}
for name, (text, kw) in demos.items():
    anim = eng.animate_text(text, fps=30, seed=3)
    render_video(anim, OUT / f"{name}.mp4", size=(512, 512), label=emo_label(anim), **kw)
    print("wrote", OUT / f"{name}.mp4", f"{anim.n / anim.fps:.1f}s")
