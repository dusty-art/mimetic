"""Video -> per-frame dense face tracks (MediaPipe Face Landmarker) + 16 kHz audio.

Each clip becomes one compressed .npz with
  lmk      (T, 478, 3) float16  normalised image coords (x, y in [0,1], z ~ x scale)
  bs       (T, 52)     float16  MediaPipe blendshape scores (ARKit names)
  mat      (T, 4, 4)   float32  canonical-face -> camera transform
  ok       (T,)        bool     face found
  audio    (S,)        int16    16 kHz mono
  fps, width, height, plus whatever metadata the dataset adapter passes.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from mimetic.paths import ASSETS

_LANDMARKER = None
BLENDSHAPE_NAMES: list[str] = []


def _landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ASSETS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4, min_tracking_confidence=0.4)
        _LANDMARKER = FaceLandmarker.create_from_options(opts)
        _LANDMARKER._ts = 0
    return _LANDMARKER


def read_audio(path: str | Path, sr: int = 16000) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(out, np.int16).copy()


def track_video(path: str | Path, max_side: int = 720) -> dict:
    import mediapipe as mp
    lm = _landmarker()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    lmks, bss, mats, oks = [], [], [], []
    W = H = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        s = max_side / max(H, W)
        if s < 1:
            frame = cv2.resize(frame, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        lm._ts += int(round(1000 / fps)) or 1
        res = lm.detect_for_video(img, lm._ts)
        if res.face_landmarks:
            p = res.face_landmarks[0]
            lmks.append(np.array([[q.x, q.y, q.z] for q in p], np.float32))
            bss.append(np.array([c.score for c in res.face_blendshapes[0]], np.float32))
            mats.append(np.asarray(res.facial_transformation_matrixes[0], np.float32))
            oks.append(True)
            if not BLENDSHAPE_NAMES:
                BLENDSHAPE_NAMES.extend(c.category_name for c in res.face_blendshapes[0])
        else:
            lmks.append(np.zeros((478, 3), np.float32))
            bss.append(np.zeros(52, np.float32))
            mats.append(np.eye(4, dtype=np.float32))
            oks.append(False)
    cap.release()
    # Reset the tracker between clips so temporal smoothing never bridges two videos.
    lm._ts += 10_000
    return {
        "lmk": np.array(lmks, np.float16) if lmks else np.zeros((0, 478, 3), np.float16),
        "bs": np.array(bss, np.float16) if bss else np.zeros((0, 52), np.float16),
        "mat": np.array(mats, np.float32) if mats else np.zeros((0, 4, 4), np.float32),
        "ok": np.array(oks, bool),
        "fps": np.float32(fps), "width": np.int32(W), "height": np.int32(H),
    }


Q = 16.0  # landmark quantisation: 1/16 pixel


def encode_landmarks(lmk: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised landmarks -> int16 pixel-space deltas over time (lossless to 1/16 px)."""
    px = lmk.astype(np.float32) * np.array([width, height, width], np.float32)
    q = np.round(px * Q).astype(np.int32)
    d = np.diff(q, axis=0, prepend=np.zeros_like(q[:1]))
    return np.clip(d, -32768, 32767).astype(np.int16)


def decode_landmarks(d: np.ndarray) -> np.ndarray:
    """int16 deltas -> (T, 478, 3) pixel coordinates (x, y, z*width)."""
    return np.cumsum(d.astype(np.int32), axis=0).astype(np.float32) / Q


def process_clip(video: str | Path, out: str | Path, meta: dict | None = None) -> bool:
    import io
    import soundfile as sf
    out = Path(out)
    if out.exists():
        return True
    tr = track_video(video)
    if len(tr["ok"]) == 0 or tr["ok"].mean() < 0.8:
        return False
    audio = read_audio(video)
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="FLAC")
    rec = {
        "lmk_d": encode_landmarks(tr["lmk"], int(tr["width"]), int(tr["height"])),
        "bs": tr["bs"], "mat": tr["mat"], "ok": tr["ok"],
        "fps": tr["fps"], "width": tr["width"], "height": tr["height"],
        "audio_flac": np.frombuffer(buf.getvalue(), np.uint8),
    }
    for k, v in (meta or {}).items():
        rec[k] = np.asarray(v)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rec)
    tmp.rename(out)
    return True


def load_clip(path: str | Path) -> dict:
    """Decode a processed clip: lmk (T,478,3) pixels, audio (S,) float32 16 kHz, plus metadata."""
    import io
    import soundfile as sf
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["lmk"] = decode_landmarks(d.pop("lmk_d"))
    raw = d.pop("audio_flac").tobytes()
    try:
        d["audio"], _ = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        d["audio"] = np.zeros(0, np.float32)
    return d
