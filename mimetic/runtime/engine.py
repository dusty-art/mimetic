"""Runtime engine: text (or audio) -> muscle activations -> face / blendshapes / avatar parameters.

    eng = Engine.load()
    anim = eng.animate_text("[happy:0.7] Hi! I'm driven by forty-seven muscles.")
    anim.acts      # (T, 52) muscle activations at anim.fps
    anim.vertices  # (T, 468, 3) simulated skin (lazy)
    anim.arkit     # (T, 52) ARKit blendshape weights for avatars (lazy)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from mimetic.data import phonemes as P
from mimetic.data import prosody
from mimetic.face import anatomy as A
from mimetic.model.controller import EMOTIONS, SpeechEmotionController
from mimetic.paths import CHECKPOINTS
from mimetic.runtime import life
from mimetic.runtime.face_model import load_face_model

CTRL_FPS = 30.0


def resample_audio(x, sr_in, sr_out=16000):
    if sr_in == sr_out:
        return x
    t_out = np.arange(int(len(x) * sr_out / sr_in)) / sr_out
    return np.interp(t_out, np.arange(len(x)) / sr_in, x).astype(np.float32)


def phone_track(segments, n, fps=CTRL_FPS):
    ids = np.zeros(n, np.int64)
    pos = np.zeros(n, np.float32)
    dur = np.zeros(n, np.float32)
    for a, b, p in segments:
        i0, i1 = int(round(a * fps)), max(int(round(a * fps)) + 1, int(round(b * fps)))
        i0, i1 = max(0, i0), min(n, i1)
        if i1 <= i0:
            continue
        ids[i0:i1] = P.PID[p]
        pos[i0:i1] = (np.arange(i0, i1) + 0.5 - a * fps) / max((b - a) * fps, 1e-3)
        dur[i0:i1] = b - a
    return ids, np.clip(pos, 0, 1), dur


def emotion_track(spans, n, fps=CTRL_FPS, ramp_s=0.35):
    """Per-frame emotion id and intensity, with intensity ramps at emotion changes."""
    emo = np.zeros(n, np.int64)
    inten = np.zeros(n, np.float32)
    for a, b, e, i in spans:
        i0, i1 = int(round(a * fps)), int(round(b * fps))
        emo[max(0, i0):min(n, i1)] = EMOTIONS.index(e) if e in EMOTIONS else 0
        inten[max(0, i0):min(n, i1)] = i
    if spans:
        last = spans[-1]
        emo[int(round(last[1] * fps)):] = EMOTIONS.index(last[2]) if last[2] in EMOTIONS else 0
        inten[int(round(last[1] * fps)):] = last[3]
    # Ramp: dip toward zero across a change of category, glide across a change of level.
    r = max(1, int(ramp_s * fps))
    ker = np.ones(r) / r
    sm = np.convolve(np.pad(inten, (r, r), mode="edge"), ker, "same")[r:-r]
    change = np.zeros(n, bool)
    change[1:] = emo[1:] != emo[:-1]
    for t in np.nonzero(change)[0]:
        lo, hi = max(0, t - r), min(n, t + r)
        w = np.abs(np.arange(lo, hi) - t) / r
        sm[lo:hi] *= w
    return emo, sm.astype(np.float32)


@dataclass
class Animation:
    fps: float
    acts: np.ndarray            # (T, N_ACT)
    head: np.ndarray            # (T, 3) pitch/yaw/roll radians
    gaze: np.ndarray            # (T, 2) yaw/pitch radians
    audio: np.ndarray | None = None
    sr: int = 24000
    segments: list = field(default_factory=list)
    emotion_spans: list = field(default_factory=list)
    engine: object = None
    _verts: np.ndarray | None = None
    _arkit: np.ndarray | None = None

    @property
    def vertices(self):
        if self._verts is None:
            self._verts = self.engine.vertices(self.acts)
        return self._verts

    @property
    def arkit(self):
        if self._arkit is None:
            self._arkit = self.engine.to_arkit(self.acts, self.gaze)
        return self._arkit

    @property
    def n(self):
        return len(self.acts)


class Engine:
    def __init__(self, controller, face, speakers, device="cpu", arkit_map=None, tts=None):
        self.ctrl = controller.eval()
        self.face = face
        self.speakers = speakers
        self.device = device
        self.arkit_map = arkit_map
        self._tts = tts
        self.n_avg = len(speakers)
        self.pal = [A.MUSCLE_NAMES.index(f"orbicularis_oculi_palpebral_{s}") for s in "RL"]
        self.lev = [A.MUSCLE_NAMES.index(f"levator_palpebrae_{s}") for s in "RL"]

    @classmethod
    def load(cls, name=None, device=None):
        import os
        name = name or os.environ.get("MIMETIC_CONTROLLER", "controller")
        device = device or "cpu"
        ck = torch.load(CHECKPOINTS / f"{name}.pt", map_location="cpu", weights_only=False)
        ctrl = SpeechEmotionController(n_speakers=ck["n_speakers"])
        ctrl.load_state_dict(ck["model"])
        face = load_face_model(device)
        arkit = None
        p = CHECKPOINTS / "arkit_map.pt"
        if p.exists():
            from mimetic.runtime.outputs import ArkitMap
            arkit = ArkitMap.load(p, device)
        return cls(ctrl.to(device), face, ck["speakers"], device, arkit)

    @property
    def tts(self):
        if self._tts is None:
            from mimetic.runtime.tts import TextToSpeech
            self._tts = TextToSpeech()
        return self._tts

    # ------------------------------------------------------------------
    @torch.no_grad()
    def drive(self, segments, emotion_spans, audio16k, n, style=None, speech_gain=1.0, seed=0, emotion_gain=1.0):
        ids, pos, dur = phone_track(segments, n)
        pr = prosody.features(audio16k, fps=CTRL_FPS, n=n) if audio16k is not None and len(audio16k) else \
            np.zeros((n, 4), np.float32)
        emo, inten = emotion_track(emotion_spans, n)
        spk = self.n_avg if style is None else (self.speakers.index(style) if isinstance(style, str) else style)
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.device)[None]
        acts, head = self.ctrl(t(ids, torch.long), t(pos), t(dur), t(pr), t(emo, torch.long), t(inten),
                               torch.tensor([spk], device=self.device), speech_gain=speech_gain,
                               emotion_gain=emotion_gain)
        acts, head = acts[0].cpu().numpy(), head[0].cpu().numpy()
        rng = np.random.default_rng(seed)
        main_emo = EMOTIONS[int(np.bincount(emo).argmax())] if n else "neutral"
        blink = life.blink_track(n, CTRL_FPS, main_emo, pause_mask=ids == 0, rng=rng)
        acts[:, self.pal] = np.maximum(acts[:, self.pal], blink[:, None] * 0.95)
        acts[:, self.lev] *= (1 - blink[:, None])
        head = head + life.head_drift(n, CTRL_FPS, rng)
        gaze = life.gaze_track(n, CTRL_FPS, rng)
        return acts, head, gaze

    def animate_text(self, text, voice=None, style=None, fps=60.0, speech_gain=1.0, seed=0,
                     default_emotion=("neutral", 0.0), emotion_gain=1.0) -> Animation:
        utt = self.tts.synthesize(text, voice=voice, default_emotion=default_emotion)
        n = int(np.ceil(utt.duration * CTRL_FPS)) + 3
        a16 = resample_audio(utt.audio, 24000)
        acts, head, gaze = self.drive(utt.segments, utt.emotion_spans, a16, n, style, speech_gain, seed, emotion_gain)
        anim = Animation(CTRL_FPS, acts, head, gaze, utt.audio, 24000, utt.segments, utt.emotion_spans, self)
        return upsample(anim, fps) if fps != CTRL_FPS else anim

    def animate_audio(self, audio, sr, text=None, emotion=("neutral", 0.0), fps=60.0, seed=0, aligner=None):
        """Offline audio file: align the transcript if given, otherwise use the audio model."""
        a16 = resample_audio(audio, sr)
        n = int(np.ceil(len(a16) / 16000 * CTRL_FPS)) + 3
        spans = [(0.0, len(a16) / 16000, emotion[0], emotion[1])]
        if text is not None:
            from mimetic.data.align import Aligner
            aligner = aligner or Aligner()
            segs = [(a, b, p) for a, b, p, _ in aligner.align(a16, text)]
            acts, head, gaze = self.drive(segs, spans, a16, n, seed=seed)
        else:
            from mimetic.runtime.live import AudioDriver
            drv = AudioDriver.load(self.device)
            acts = drv.offline(a16, emotion)
            rng = np.random.default_rng(seed)
            n = len(acts)
            head = life.head_drift(n, CTRL_FPS, rng)
            gaze = life.gaze_track(n, CTRL_FPS, rng)
            segs = []
        anim = Animation(CTRL_FPS, acts, head, gaze, audio, sr, segs, spans, self)
        return upsample(anim, fps) if fps != CTRL_FPS else anim

    @torch.no_grad()
    def vertices(self, acts):
        out = []
        for i in range(0, len(acts), 1024):
            a = torch.as_tensor(acts[i:i + 1024], dtype=torch.float32, device=self.device)
            out.append(self.face(a).cpu().numpy())
        return np.concatenate(out)

    def to_arkit(self, acts, gaze=None):
        from mimetic.runtime.outputs import arkit_from_acts
        return arkit_from_acts(self, acts, gaze)


def upsample(anim: Animation, fps: float) -> Animation:
    n_out = int(round(anim.n / anim.fps * fps))
    t_in = np.arange(anim.n) / anim.fps
    t_out = np.arange(n_out) / fps

    def interp(x):
        return np.stack([np.interp(t_out, t_in, x[:, k]) for k in range(x.shape[1])], -1).astype(np.float32)
    return Animation(fps, interp(anim.acts), interp(anim.head), interp(anim.gaze), anim.audio, anim.sr,
                     anim.segments, anim.emotion_spans, anim.engine)
