"""Behaviours that are stochastic rather than speech-driven: blinks, gaze shifts, idle head drift.

Blinks: ~17/min at rest, more under arousal, preferentially at pauses and
phrase boundaries; fast closing (~80 ms), brief closure, slower opening
(~180 ms). Gaze: fixations with small saccades every 0.6-2.5 s. Head: slow
band-limited drift on top of the controller's prosodic head motion.
"""
from __future__ import annotations

import numpy as np

BLINK_RATE = {"neutral": 17, "calm": 14, "happy": 19, "sad": 15, "angry": 21, "fearful": 26, "disgusted": 18,
              "surprised": 12, "contempt": 16}


def blink_shape(fps):
    close = np.linspace(0, 1, max(2, int(0.08 * fps)), endpoint=False) ** 1.5
    hold = np.ones(max(1, int(0.04 * fps)))
    opn = 1 - np.linspace(0, 1, max(2, int(0.18 * fps))) ** 0.7
    return np.concatenate([close, hold, opn])


def blink_track(T, fps, emotion="neutral", pause_mask=None, rng=None, rate=None):
    rng = rng or np.random.default_rng()
    rate = rate or BLINK_RATE.get(emotion, 17)
    p = np.full(T, rate / 60 / fps)
    if pause_mask is not None:
        p = p * np.where(pause_mask, 3.0, 0.6)
    out = np.zeros(T)
    shape = blink_shape(fps)
    refractory = int(0.6 * fps)
    t = int(rng.integers(0, int(fps)))
    last = -refractory
    while t < T:
        if t - last > refractory and rng.random() < p[t]:
            n = min(len(shape), T - t)
            out[t:t + n] = np.maximum(out[t:t + n], shape[:n])
            last = t
            t += len(shape)
        t += 1
    return out


def gaze_track(T, fps, rng=None, amplitude=0.06):
    rng = rng or np.random.default_rng()
    g = np.zeros((T, 2))
    cur = np.zeros(2)
    t = 0
    while t < T:
        dwell = int(rng.uniform(0.6, 2.5) * fps)
        tgt = rng.normal(0, amplitude, 2) * np.array([1.0, 0.6])
        n = min(3, T - t)
        for k in range(n):                   # saccade: ~3 frames
            g[t + k] = cur + (tgt - cur) * (k + 1) / n
        cur = tgt
        g[t + n:t + dwell] = cur
        t += dwell
    return g


def head_drift(T, fps, rng=None, amp=(0.025, 0.035, 0.012)):
    """Band-limited noise (pitch, yaw, roll) in radians."""
    rng = rng or np.random.default_rng()
    out = np.zeros((T, 3))
    for k in range(3):
        n = rng.normal(0, 1, T + 200)
        ker = np.exp(-0.5 * (np.arange(-90, 91) / (0.9 * fps)) ** 2)
        s = np.convolve(n, ker / ker.sum(), "same")[100:100 + T]
        out[:, k] = s / (s.std() + 1e-9) * amp[k]
    return out
