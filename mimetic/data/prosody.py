"""Frame-rate prosody features: loudness (dB) and pitch (F0) with voicing, from 16 kHz audio."""
from __future__ import annotations

import numpy as np


def frame_signal(x: np.ndarray, sr: int, fps: float, win_s: float):
    hop = sr / fps
    n = int(np.ceil(len(x) / hop))
    win = int(win_s * sr)
    pad = np.pad(x, (win // 2, win + int(hop) + 1))
    idx = (np.arange(n)[:, None] * hop).astype(np.int64) + np.arange(win)[None]
    return pad[idx]


def loudness_db(x, sr=16000, fps=30.0):
    fr = frame_signal(x, sr, fps, 0.03)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-10)
    return 20 * np.log10(rms + 1e-5)


def pitch(x, sr=16000, fps=30.0, fmin=65.0, fmax=420.0):
    """Normalised-autocorrelation pitch tracker. Returns f0 (Hz, 0 when unvoiced) and voicing strength."""
    fr = frame_signal(x, sr, fps, 0.045)
    fr = fr - fr.mean(1, keepdims=True)
    fr *= np.hanning(fr.shape[1])[None]
    n = fr.shape[1]
    F = np.fft.rfft(fr, 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, axis=1)[:, :n]
    ac /= ac[:, :1] + 1e-9
    lo, hi = int(sr / fmax), int(sr / fmin)
    seg = ac[:, lo:hi]
    k = seg.argmax(1)
    strength = seg[np.arange(len(seg)), k]
    # parabolic interpolation of the peak
    k0 = np.clip(k, 1, seg.shape[1] - 2)
    a, b, c = seg[np.arange(len(seg)), k0 - 1], seg[np.arange(len(seg)), k0], seg[np.arange(len(seg)), k0 + 1]
    off = 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
    lag = lo + k0 + np.clip(off, -1, 1)
    f0 = sr / lag
    db = loudness_db(x, sr, fps)[: len(f0)]
    voiced = (strength > 0.5) & (db > db.max() - 35)
    return np.where(voiced, f0, 0.0), strength


def features(x, sr=16000, fps=30.0, n=None):
    """-> (T, 4): loudness (dB re clip max), voiced, log-F0 (semitones re clip median), voicing strength."""
    db = loudness_db(x, sr, fps)
    f0, st = pitch(x, sr, fps)
    T = min(len(db), len(f0)) if n is None else n
    db = np.resize(db, T) if len(db) < T else db[:T]
    f0 = np.resize(f0, T) if len(f0) < T else f0[:T]
    st = np.resize(st, T) if len(st) < T else st[:T]
    v = f0 > 0
    semis = np.zeros(T)
    if v.sum() > 3:
        med = np.median(f0[v])
        semis[v] = 12 * np.log2(f0[v] / med)
    return np.stack([np.clip(db - db.max(), -60, 0) / 60 + 1, v.astype(float), np.clip(semis, -12, 12) / 12,
                     st], -1).astype(np.float32)
