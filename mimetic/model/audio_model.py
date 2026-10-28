"""Streaming audio -> muscle activations ("the ear") for live microphone use.

Causal dilated convolutions over 10 ms log-mel frames; the output is delayed by
`lookahead` frames, so the model sees that much future audio. Lips move before
the sound they shape (rounding leads by ~100 ms), and a short lookahead is
what lets a causal model anticipate it. Total latency = lookahead + one hop.

Auxiliary heads (phoneme posteriors, emotion, voice activity) shape the
representation and let the live UI show what the model hears.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimetic.data import phonemes as P
from mimetic.face import anatomy as A
from mimetic.model.controller import EMOTIONS

SR = 16000
HOP = 160
N_FFT = 400
N_MELS = 80


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=50.0, fmax=7600.0):
    def hz2mel(f): return 2595 * math.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    mels = torch.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = torch.tensor([mel2hz(m.item()) for m in mels])
    bins = torch.floor((n_fft + 1) * hz / sr).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        a, b, c = bins[i].item(), bins[i + 1].item(), bins[i + 2].item()
        for k in range(a, b):
            fb[i, k] = (k - a) / max(1, b - a)
        for k in range(b, c):
            fb[i, k] = (c - k) / max(1, c - b)
    return fb


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fb", mel_filterbank())
        self.register_buffer("win", torch.hann_window(N_FFT))

    def forward(self, wav):
        """wav (B, S) float in [-1,1] -> (B, T, 80) log-mel at 100 Hz (frame t covers samples ~[t*160, t*160+400))."""
        spec = torch.stft(wav, N_FFT, HOP, window=self.win, center=False, return_complex=True).abs() ** 2
        mel = torch.einsum("mf,bft->btm", self.fb, spec)
        return torch.log(mel + 1e-6)


class FrameNorm(nn.Module):
    """LayerNorm over channels at each time step (causal, unlike GroupNorm over time)."""

    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class CausalBlock(nn.Module):
    def __init__(self, ch, dilation, k=3, dropout=0.1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(ch, 2 * ch, k, dilation=dilation)
        self.proj = nn.Conv1d(ch, ch, 1)
        self.norm = FrameNorm(ch)
        self.drop = nn.Dropout(dropout)

    def _gate(self, y):
        a, b = self.conv(y).chunk(2, 1)
        return self.proj(torch.tanh(a) * torch.sigmoid(b))

    def forward(self, x):
        return x + self.drop(self._gate(F.pad(self.norm(x), (self.pad, 0))))

    def stream(self, x, cache):
        """x (B,C,n) new frames; cache (B,C,pad) previous normalised inputs -> (out, new cache)."""
        y = torch.cat([cache, self.norm(x)], 2)
        return x + self._gate(y), y[:, :, -self.pad:] if self.pad else cache

    @property
    def receptive(self):
        return self.pad


class AudioToMuscle(nn.Module):
    def __init__(self, ch=256, dilations=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32), lookahead=8,
                 n_act=A.N_ACT):
        super().__init__()
        self.lookahead = lookahead
        self.mel = LogMel()
        self.inp = nn.Conv1d(N_MELS, ch, 1)
        self.emo_emb = nn.Embedding(len(EMOTIONS) + 1, ch)        # last index = "infer from voice"
        self.blocks = nn.ModuleList([CausalBlock(ch, d) for d in dilations])
        self.act_head = nn.Conv1d(ch, n_act, 1)
        self.phone_head = nn.Conv1d(ch, P.N_PHONES, 1)
        self.emo_head = nn.Conv1d(ch, len(EMOTIONS), 1)
        self.vad_head = nn.Conv1d(ch, 1, 1)
        self.register_buffer("mel_mean", torch.zeros(N_MELS))
        self.register_buffer("mel_std", torch.ones(N_MELS))
        nn.init.constant_(self.act_head.bias, -3.0)

    @property
    def receptive_field(self):
        return 1 + sum(b.receptive for b in self.blocks)

    def forward_mel(self, mel, emotion, intensity):
        """mel (B,T,80); emotion (B,) id (len(EMOTIONS) = infer); intensity (B,). Outputs at mel rate."""
        x = ((mel - self.mel_mean) / self.mel_std).transpose(1, 2)
        h = self.inp(x)
        e = self.emo_emb(emotion) * torch.where(emotion == len(EMOTIONS), torch.ones_like(intensity), intensity)[:, None]
        h = h + e[..., None]
        for b in self.blocks:
            h = b(h)
        return {"acts": torch.sigmoid(self.act_head(h)).transpose(1, 2),
                "phone": self.phone_head(h).transpose(1, 2),
                "emotion": self.emo_head(h).transpose(1, 2),
                "vad": self.vad_head(h).transpose(1, 2)[..., 0]}

    def forward(self, wav, emotion, intensity):
        return self.forward_mel(self.mel(wav), emotion, intensity)

    def init_state(self, B=1, device="cpu"):
        return [torch.zeros(B, self.inp.out_channels, b.pad, device=device) for b in self.blocks]

    def stream(self, mel, emotion, intensity, state):
        """Incremental inference: mel (B,n,80) new frames -> outputs for those frames, new state."""
        x = ((mel - self.mel_mean) / self.mel_std).transpose(1, 2)
        h = self.inp(x)
        e = self.emo_emb(emotion) * torch.where(emotion == len(EMOTIONS), torch.ones_like(intensity), intensity)[:, None]
        h = h + e[..., None]
        new = []
        for blk, c in zip(self.blocks, state):
            h, c2 = blk.stream(h, c)
            new.append(c2)
        return {"acts": torch.sigmoid(self.act_head(h)).transpose(1, 2),
                "phone": self.phone_head(h).transpose(1, 2),
                "emotion": self.emo_head(h).transpose(1, 2),
                "vad": self.vad_head(h).transpose(1, 2)[..., 0]}, new
