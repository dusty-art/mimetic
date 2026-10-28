"""Text -> speech audio + exact phoneme timing (Kokoro-82M, Apache-2.0), with inline emotion tags.

Tags set the emotion from that point on:  "[happy:0.8] We won! [sad] But only just."
Kokoro's duration predictor gives a duration for every phoneme character, so
the face is driven by the same timing the voice was synthesised with: no
alignment error at all on the text path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from mimetic.data import phonemes as P

TAG = re.compile(r"\[(\w+)(?::([0-9.]+))?\]")
SR = 24000
FRAME_S = 600 / SR          # Kokoro duration unit (25 ms)

# Speaking-rate tendencies of emotional speech (relative to neutral).
EMOTION_RATE = {"neutral": 1.0, "calm": 0.95, "happy": 1.05, "sad": 0.88, "angry": 1.06, "fearful": 1.1,
                "disgusted": 0.95, "surprised": 1.04, "contempt": 0.97}


@dataclass
class Utterance:
    audio: np.ndarray                       # float32, 24 kHz
    segments: list                          # [(start_s, end_s, phone)]
    emotion_spans: list                     # [(start_s, end_s, emotion, intensity)]
    words: list = field(default_factory=list)   # [(start_s, end_s, text)]

    @property
    def duration(self):
        return len(self.audio) / SR


def parse_tags(text: str, default=("neutral", 0.0)):
    """-> [(chunk_text, emotion, intensity)]"""
    out, emo = [], default
    pos = 0
    for m in TAG.finditer(text):
        chunk = text[pos:m.start()].strip()
        if chunk:
            out.append((chunk, *emo))
        name = m.group(1).lower()
        name = {"joy": "happy", "anger": "angry", "fear": "fearful", "disgust": "disgusted",
                "surprise": "surprised", "sadness": "sad"}.get(name, name)
        inten = float(m.group(2)) if m.group(2) else (0.0 if name == "neutral" else 0.8)
        emo = (name, inten)
        pos = m.end()
    chunk = text[pos:].strip()
    if chunk:
        out.append((chunk, *emo))
    return out


def kokoro_segments(phonemes: str, pred_dur, t0: float):
    """Kokoro phoneme string + per-token durations (BOS/EOS included) -> unified phone segments."""
    d = np.asarray(pred_dur, np.float64)[1:-1] * FRAME_S
    t = t0 + float(pred_dur[0]) * FRAME_S
    segs = []
    carry = 0.0
    for ch, dur in zip(phonemes, d):
        ph = P.misaki_to_unified(ch)
        if ph is None:           # stress/length marks: give their time to the next phone
            carry += dur
            continue
        dur += carry
        carry = 0.0
        if segs and segs[-1][2] == ph == "SIL":
            segs[-1] = (segs[-1][0], t + dur, "SIL")
        else:
            segs.append((t, t + dur, ph))
        t += dur
    return merge_short_silences(segs)


def merge_short_silences(segs, min_sil=0.12):
    """Word gaps in TTS output are a frame or two long; in natural speech those are coarticulated, not pauses."""
    out = []
    for i, (a, b, p) in enumerate(segs):
        if p == "SIL" and b - a < min_sil and out and i + 1 < len(segs):
            mid = (a + b) / 2
            out[-1] = (out[-1][0], mid, out[-1][2])
            nxt = segs[i + 1]
            segs[i + 1] = (mid, nxt[1], nxt[2])
            continue
        out.append((a, b, p))
    return out


class TextToSpeech:
    def __init__(self, voice: str = "af_heart", lang: str = "a", device: str | None = None):
        import mimetic  # noqa: F401
        from kokoro import KPipeline
        self.pipe = KPipeline(lang_code=lang, device=device or "cpu")
        self.voice = voice

    def synthesize(self, text: str, voice: str | None = None, default_emotion=("neutral", 0.0),
                   rate: float = 1.0) -> Utterance:
        voice = voice or self.voice
        audio, segs, spans, words = [], [], [], []
        t = 0.0
        for chunk, emo, inten in parse_tags(text, default_emotion):
            speed = rate * (1 + (EMOTION_RATE.get(emo, 1.0) - 1) * min(1.0, inten / 0.8))
            start = t
            for r in self.pipe(chunk, voice=voice, speed=speed):
                if r.audio is None:
                    continue
                a = r.audio.detach().cpu().numpy().astype(np.float32)
                segs += kokoro_segments(r.phonemes, r.pred_dur.cpu().numpy(), t)
                for tok in r.tokens or []:
                    if tok.start_ts is not None and tok.end_ts is not None:
                        words.append((t + tok.start_ts, t + tok.end_ts, tok.text))
                audio.append(a)
                t += len(a) / SR
            spans.append((start, t, emo, inten))
        a = np.concatenate(audio) if audio else np.zeros(0, np.float32)
        return Utterance(a, segs, spans, words)
