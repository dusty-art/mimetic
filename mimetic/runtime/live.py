"""Live microphone -> muscles -> avatar, in real time on CPU.

Audio arrives in 10 ms hops. The driver keeps the model's receptive field of
past log-mel frames and runs one step per hop batch; the emitted face frame
corresponds to `lookahead` hops in the past (default 80 ms).
"""
from __future__ import annotations

import numpy as np
import torch

from mimetic.data import phonemes as P
from mimetic.face import anatomy as A
from mimetic.model.audio_model import HOP, N_FFT, SR, AudioToMuscle
from mimetic.model.controller import EMOTIONS
from mimetic.paths import CHECKPOINTS
from mimetic.runtime import life


class LivePost:
    """Causal post-processing of the audio model's 100 Hz muscle frames, shared by streaming and
    offline inference so the two stay identical.

    * Emotion pathway (emotion "auto" only): the voice-emotion posterior, sharpened and smoothed over
      speech, adds the mocap's per-emotion muscle offsets in logit space, the same composition the
      text controller uses. The lip sphincter and jaw are spared, and while the voice says p/b/m/f/v the
      offsets on lip-parting muscles yield (released over 60 ms), the rule the text controller learned:
      a smile gives way to a lip seal but keeps the cheeks and eyes.
    * One Euro filter (Casiez et al. 2012) per muscle: strong smoothing while a muscle holds still,
      little while it moves fast, so jitter goes without lagging quick gestures.
    """

    PATH = CHECKPOINTS / "live_post.npz"
    LABIALS = [P.PID[p] for p in ("P", "B", "M", "F", "V")]
    LIP_PARTING = np.array([any(n.startswith(m) for m in (
        "levator_labii", "zygomaticus", "levator_anguli", "risorius", "buccinator", "depressor_anguli",
        "depressor_labii", "platysma")) for n in A.MUSCLE_NAMES], np.float32)

    def __init__(self, offsets, min_cutoff=8.0, beta=2.0, d_cutoff=1.0, emo_tau=0.8, sharpen=1.0, gain=1.0,
                 rate=100.0):
        self.offsets = np.asarray(offsets, np.float32)                 # (N_EMOTIONS, N_ACT) logit offsets
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.emo_tau, self.sharpen, self.gain, self.rate = float(emo_tau), float(sharpen), float(gain), rate
        self.k_emo = 1 - np.exp(-1 / (emo_tau * rate))
        self.release = np.exp(-1 / (0.06 * rate))
        self.reset()

    @classmethod
    def load(cls, **kw):
        if not cls.PATH.exists():
            return None
        z = np.load(cls.PATH)
        cfg = {k: float(z[k]) for k in ("min_cutoff", "beta", "emo_tau", "sharpen", "gain") if k in z.files}
        return cls(z["offsets"], **{**cfg, **kw})

    def fresh(self):
        return LivePost(self.offsets, self.min_cutoff, self.beta, self.d_cutoff, self.emo_tau, self.sharpen,
                        self.gain, self.rate)

    def reset(self):
        self.x = self.dx = None
        self.p = np.eye(len(self.offsets), dtype=np.float32)[0]       # start neutral
        self.lab = 0.0

    def _alpha(self, cutoff):
        tau = 1 / (2 * np.pi * cutoff)
        return 1 / (1 + tau * self.rate)

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    def __call__(self, acts, emo_logits=None, vad=None, gain=0.0, phone_logits=None):
        out = np.empty_like(acts)
        if emo_logits is not None:
            pe = self._softmax(emo_logits * self.sharpen)
            lab = (self._softmax(phone_logits)[:, self.LABIALS].sum(-1) if phone_logits is not None
                   else np.zeros(len(acts)))
        ad = self._alpha(self.d_cutoff)
        for t, a in enumerate(acts):
            if gain and emo_logits is not None:
                self.p = self.p + self.k_emo * float(vad[t]) * (pe[t] - self.p)
                self.lab = max(float(lab[t]), self.lab * self.release)
                z = np.log(np.clip(a, 1e-3, 1 - 1e-3) / np.clip(1 - a, 1e-3, 1))
                off = (self.p @ self.offsets) * (1 - min(1.0, 1.5 * self.lab) * self.LIP_PARTING)
                a = 1 / (1 + np.exp(-(z + self.gain * gain * off)))
            if self.x is None:
                self.x, self.dx = a.copy(), np.zeros_like(a)
            else:
                self.dx = self.dx + ad * ((a - self.x) * self.rate - self.dx)
                al = self._alpha(self.min_cutoff + self.beta * np.abs(self.dx))
                self.x = self.x + al * (a - self.x)
            out[t] = self.x
        return out


class AudioDriver:
    def __init__(self, model: AudioToMuscle, device="cpu", post: LivePost | None = None):
        self.m = model.eval().to(device)
        self.device = device
        self.rf = model.receptive_field
        self.post = post
        self.reset()

    @classmethod
    def load(cls, device="cpu", name="audio_model", post=True):
        ck = torch.load(CHECKPOINTS / f"{name}.pt", map_location="cpu", weights_only=False)
        m = AudioToMuscle(**ck.get("config", {}))
        m.load_state_dict(ck["model"])
        return cls(m, device, LivePost.load() if post else None)

    def reset(self):
        self.buf = np.zeros(0, np.float32)
        self.state = self.m.init_state(1, self.device)
        self.n_out = 0
        if self.post is not None:
            self.post.reset()

    @staticmethod
    def _post_inputs(out, emotion, sl):
        if emotion[0] != "auto":
            return None, None, 0.0, None
        return (out["emotion"][0, sl].cpu().numpy(), torch.sigmoid(out["vad"][0, sl]).cpu().numpy(),
                float(emotion[1]), out["phone"][0, sl].cpu().numpy())

    @torch.no_grad()
    def raw(self, audio16k, emotion=("neutral", 0.0)):
        """Whole-file model outputs at the 100 Hz hop rate, lookahead removed."""
        wav = torch.as_tensor(audio16k, dtype=torch.float32, device=self.device)[None]
        wav = torch.nn.functional.pad(wav, (0, HOP * (self.m.lookahead + 2)))
        e = torch.tensor([EMOTIONS.index(emotion[0]) if emotion[0] != "auto" else len(EMOTIONS)], device=self.device)
        return self.m(wav, e, torch.tensor([float(emotion[1])], device=self.device))

    @torch.no_grad()
    def offline(self, audio16k, emotion=("neutral", 0.0), out_fps=30.0, return_phones=False, post=None):
        """Whole-file inference; returns (T, N_ACT) at out_fps, aligned to the audio (lookahead removed).
        With return_phones, also the phone posteriors (T, N_PHONES) that drive the tongue.
        `post` overrides the driver's post-processor (False disables it)."""
        out = self.raw(audio16k, emotion)
        sl = slice(self.m.lookahead, None)
        acts = out["acts"][0, sl].cpu().numpy()                        # undo the output delay
        post = self.post if post is None else (post or None)
        if post is not None:
            acts = post.fresh()(acts, *self._post_inputs(out, emotion, sl))
        t_in = (np.arange(len(acts)) * HOP + N_FFT / 2) / SR
        n_out = int(np.ceil(len(audio16k) / SR * out_fps))
        t_out = np.arange(n_out) / out_fps
        res = lambda x: np.stack([np.interp(t_out, t_in, x[:, k]) for k in range(x.shape[1])], -1).astype(np.float32)
        if return_phones:
            ph = torch.softmax(out["phone"][0, sl], -1).cpu().numpy()
            return res(acts), res(ph)
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
        if self.post is not None:
            # The first `lookahead` frames precede the audio (offline drops them); keep them out of the
            # post-processor's state so streaming and offline stay identical.
            k = min(len(acts), max(0, self.m.lookahead - self.n_out))
            acts[k:] = self.post(acts[k:], *self._post_inputs(out, emotion, slice(k, None)))
        self.n_out += len(acts)
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
