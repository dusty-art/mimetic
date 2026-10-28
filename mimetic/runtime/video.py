"""Render an Animation to an MP4 with its audio (headless OpenGL + ffmpeg)."""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from mimetic.face.render import FaceRenderer
from mimetic.face.sim import FaceSim
from mimetic.runtime.export3d import jaw_tracks

_R = {}


def renderer(size):
    key = tuple(size)
    if key not in _R:
        sim = FaceSim(calibrate=False)
        _R[key] = FaceRenderer(sim, size)
    return _R[key]


def head_matrix(h):
    p, y, r = h
    cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    M = np.eye(4, dtype=np.float32)
    piv = np.array([0, -2.0, -3.0])        # rotate about the upper neck, not the nose
    R = Ry @ Rx @ Rz
    M[:3, :3] = R
    M[:3, 3] = piv - R @ piv
    return M


_FONT = None


def _caption(img, text):
    from PIL import Image, ImageDraw, ImageFont
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.load_default(size=max(14, img.shape[0] // 26))
        except TypeError:
            _FONT = ImageFont.load_default()
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = d.textbbox((14, 12), text, font=_FONT)
    d.rounded_rectangle([x0 - 8, y0 - 6, x1 + 8, y1 + 6], radius=6, fill=(255, 255, 255))
    d.text((14, 12), text, fill=(20, 20, 20), font=_FONT)
    return np.asarray(im)


def render_video(anim, out, size=(512, 512), fps=None, show_muscles=False, yaw=0.0, label=None):
    fps = fps or anim.fps
    r = renderer(size)
    r.set_camera(yaw=yaw)
    n_out = int(round(anim.n / anim.fps * fps))
    idx = np.minimum((np.arange(n_out) * anim.fps / fps).round().astype(int), anim.n - 1)
    verts = anim.vertices
    jaws = jaw_tracks(anim.acts)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        if anim.audio is not None:
            import soundfile as sf
            sf.write(wav, anim.audio, anim.sr)
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
               "-r", str(fps), "-i", "-"]
        if anim.audio is not None:
            cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "128k"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-shortest", str(out)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for k in idx:
            J = jaws[k]
            img = r.render(verts[k], act=anim.acts[k], jaw=(J[:3, :3], J[:3, 3]), head=head_matrix(anim.head[k]),
                           gaze=tuple(anim.gaze[k]), show_muscles=show_muscles,
                           skin_alpha=0.7 if show_muscles else 1.0)
            if label is not None:
                img = _caption(img, label(k / anim.fps) if callable(label) else label)
            p.stdin.write(img.tobytes())
        p.stdin.close()
        p.wait()
    return out
