"""Live microphone -> muscles -> avatar, in real time on CPU.

Audio arrives in 10 ms hops. The driver keeps the model's receptive field of
past log-mel frames and runs one step per hop batch; the emitted face frame
corresponds to `lookahead` hops in the past (default 80 ms).
"""
from __future__ import annotations

import numpy as np
import torch

from mimetic.face import anatomy as A
from mimetic.model.audio_model import HOP, N_FFT, SR, AudioToMuscle
from mimetic.model.controller import EMOTIONS
from mimetic.paths import CHECKPOINTS
from mimetic.runtime import life


class AudioDriver:
    def __init__(self, model: AudioToMuscle, device="cpu"):
        self.m = model.eval().to(device)
        self.device = device
        self.rf = model.receptive_field
        self.reset()

    @classmethod
    def load(cls, device="cpu", name="audio_model"):
        ck = torch.load(CHECKPOINTS / f"{name}.pt", map_location="cpu", weights_only=False)
        m = AudioToMuscle(**ck.get("config", {}))
        m.load_state_dict(ck["model"])
        return cls(m, device)

    def reset(self):
        self.buf = np.zeros(0, np.float32)
        self.state = self.m.init_state(1, self.device)

    @torch.no_grad()
    def offline(self, audio16k, emotion=("neutral", 0.0), out_fps=30.0, return_phones=False):
        """Whole-file inference; returns (T, N_ACT) at out_fps, aligned to the audio (lookahead removed).
        With return_phones, also the phone posteriors (T, N_PHONES) that drive the tongue."""
        wav = torch.as_tensor(audio16k, dtype=torch.float32, device=self.device)[None]
        wav = torch.nn.functional.pad(wav, (0, HOP * (self.m.lookahead + 2)))
        e = torch.tensor([EMOTIONS.index(emotion[0]) if emotion[0] != "auto" else len(EMOTIONS)], device=self.device)
        out = self.m(wav, e, torch.tensor([float(emotion[1])], device=self.device))
        acts = out["acts"][0, self.m.lookahead:].cpu().numpy()      # undo the output delay
        t_in = (np.arange(len(acts)) * HOP + N_FFT / 2) / SR
        n_out = int(np.ceil(len(audio16k) / SR * out_fps))
        t_out = np.arange(n_out) / out_fps
        res = lambda x: np.stack([np.interp(t_out, t_in, x[:, k]) for k in range(x.shape[1])], -1).astype(np.float32)
        if return_phones:
            post = torch.softmax(out["phone"][0, self.m.lookahead:], -1).cpu().numpy()
            return res(acts), res(post)
        return res(acts)

    @torch.no_grad()
    def push(self, chunk16k: np.ndarray, emotion=("auto", 1.0)):
        """Feed new samples; returns list of activation frames (one per 10 ms hop)."""
        self.buf = np.concatenate([self.buf, chunk16k.astype(np.float32)])
        frames = []
        while len(self.buf) >= N_FFT:
            frames.append(self.buf[:N_FFT].copy())
            self.buf = self.buf[HOP:]
        if not frames:
            return []
        wav = torch.as_tensor(np.stack(frames), device=self.device)
        mel = self.m.mel(wav)[:, 0][None]                              # (1, n, 80): one mel frame per hop
        e = torch.tensor([EMOTIONS.index(emotion[0]) if emotion[0] != "auto" else len(EMOTIONS)], device=self.device)
        out, self.state = self.m.stream(mel, e, torch.tensor([float(emotion[1])], device=self.device), self.state)
        acts = out["acts"][0].cpu().numpy()
        from mimetic.face.tongue import CRIT, TARGETS
        post = torch.softmax(out["phone"][0], -1).cpu().numpy()
        # Live tongue: first-order approach to the posterior-weighted articulatory target.
        if not hasattr(self, "tongue"):
            self.tongue = np.zeros(5, np.float32)
        tongues = []
        for p in post:
            k = 0.25 * (0.6 + 0.8 * float(p @ CRIT))
            self.tongue = self.tongue + k * (p @ TARGETS - self.tongue)
            tongues.append(self.tongue.copy())
        self.last_tongue = tongues
        self.last_emotion = torch.softmax(out["emotion"][0, -1], -1).cpu().numpy()
        self.last_vad = torch.sigmoid(out["vad"][0, -1]).item()
        return list(acts)


class LiveSession:
    """Microphone capture -> AudioDriver -> callbacks with (acts, head, gaze) at ~100 Hz."""

    def __init__(self, engine, driver: AudioDriver, on_frame, emotion=("auto", 1.0), device_index=None):
        self.engine, self.drv, self.on_frame = engine, driver, on_frame
        self.emotion = emotion
        self.device_index = device_index
        self.running = False
        self.rng = np.random.default_rng()
        self.blink_phase = None
        self.t = 0

    def start(self):
        import sounddevice as sd
        self.running = True
        pal = [A.MUSCLE_NAMES.index(f"orbicularis_oculi_palpebral_{s}") for s in "RL"]
        shape = life.blink_shape(100.0)
        head_noise = life.head_drift(100 * 600, 100.0, self.rng)
        gaze = life.gaze_track(100 * 600, 100.0, self.rng)
        state = {"blink": -1}

        def cb(indata, frames, timeinfo, status):
            mono = indata[:, 0].astype(np.float32)
            for a in self.drv.push(mono, self.emotion):
                i = self.t % len(head_noise)
                if state["blink"] < 0 and self.rng.random() < 17 / 60 / 100:
                    state["blink"] = 0
                if state["blink"] >= 0:
                    a[pal] = np.maximum(a[pal], shape[state["blink"]] * 0.95)
                    state["blink"] += 1
                    if state["blink"] >= len(shape):
                        state["blink"] = -1
                self.on_frame(a, head_noise[i], gaze[i])
                self.t += 1

        self.stream = sd.InputStream(samplerate=SR, channels=1, blocksize=HOP * 2, callback=cb,
                                     device=self.device_index, latency="low")
        self.stream.start()

    def stop(self):
        self.running = False
        self.stream.stop(); self.stream.close()
