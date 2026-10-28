"""Speech + emotion motor controller: what to say and how you feel -> facial muscle activations.

Two pathways meet in muscle space:
  speech trunk  : phonemes (+ articulatory features, timing), prosody, speaking style
                  -> articulation drive (logits)
  emotion trunk : emotion x intensity, prosody, and the speech trunk's state
                  -> emotional drive (logits), gated to zero at zero intensity
  activations   = sigmoid(articulation + emotion + tone)

Adding drives in logit space gives the saturation real muscles have, and
because the emotion trunk sees the speech state it learns to yield where
speech needs the mouth (a smile narrows for "oo" but keeps the eyes and
cheeks). Speech-critical articulation is enforced separately by the
articulation constraints during training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from mimetic.data import phonemes as P
from mimetic.face import anatomy as A

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "contempt", "calm"]


def sinusoid(T, d, device):
    pos = torch.arange(T, device=device)[:, None]
    i = torch.arange(0, d, 2, device=device)[None]
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class Block(nn.Module):
    def __init__(self, d, heads, causal_lookahead=None, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.conv = nn.Conv1d(d, d, 5, padding=2, groups=d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.lookahead = causal_lookahead

    def forward(self, h, pad_mask=None):
        T = h.shape[1]
        attn_mask = None
        if self.lookahead is not None:
            i = torch.arange(T, device=h.device)
            attn_mask = (i[None, :] - i[:, None]) > self.lookahead
        y = self.ln1(h)
        y, _ = self.attn(y, y, y, key_padding_mask=pad_mask, attn_mask=attn_mask, need_weights=False)
        h = h + y
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        return h + self.ff(self.ln2(h))


class SpeechEmotionController(nn.Module):
    def __init__(self, n_speakers: int, d: int = 256, layers: int = 6, d_emo: int = 160, emo_layers: int = 3,
                 style_dim: int = 8, n_act: int = A.N_ACT, heads: int = 4, lookahead: int | None = None,
                 output: str = "sigmoid"):
        super().__init__()
        self.n_act = n_act
        self.output = output          # "linear" = baseline that regresses displacement PCA directly
        self.phone_emb = nn.Embedding(P.N_PHONES, 96)
        n_in = 96 + P.ARTIC.shape[1] + 3 + 4 + style_dim
        self.register_buffer("artic", torch.tensor(P.ARTIC))
        self.inp = nn.Linear(n_in, d)
        self.blocks = nn.ModuleList([Block(d, heads, lookahead) for _ in range(layers)])
        self.speech_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_act))
        self.style = nn.Embedding(n_speakers + 1, style_dim)      # last index = average style
        nn.init.normal_(self.style.weight, std=0.1)
        with torch.no_grad():
            self.style.weight[-1].zero_()
        self.emo_emb = nn.Embedding(len(EMOTIONS), 64)
        self.emo_inp = nn.Linear(64 + 1 + 4 + d, d_emo)
        self.emo_blocks = nn.ModuleList([Block(d_emo, 4, lookahead) for _ in range(emo_layers)])
        self.emo_out = nn.Sequential(nn.LayerNorm(d_emo), nn.Linear(d_emo, n_act))
        self.head_out = nn.Sequential(nn.LayerNorm(d + d_emo), nn.Linear(d + d_emo, 3))
        self.tone = nn.Parameter(torch.full((n_act,), -3.0 if output == "sigmoid" else 0.0))
        nn.init.zeros_(self.emo_out[1].weight); nn.init.zeros_(self.emo_out[1].bias)
        nn.init.zeros_(self.speech_out[1].weight); nn.init.zeros_(self.speech_out[1].bias)

    def forward(self, phone, ppos, pdur, prosody, emotion, intensity, speaker, pad_mask=None,
                speech_gain: float = 1.0, return_parts: bool = False, emo_prosody_keep=None, emo_ctx_keep=None):
        """phone/ppos/pdur/prosody: (B,T[,4]); emotion (B,) or (B,T) ids; intensity (B,) or (B,T); speaker (B,)."""
        B, T = phone.shape
        dev = phone.device
        sty = self.style(speaker)[:, None].expand(B, T, -1)
        feats = torch.cat([self.phone_emb(phone), self.artic[phone], ppos[..., None], pdur[..., None].clamp(0, 1),
                           (phone > 0).float()[..., None], prosody, sty], -1)
        h = self.inp(feats) + sinusoid(T, self.inp.out_features, dev)[None]
        for blk in self.blocks:
            h = blk(h, pad_mask)
        speech = self.speech_out(h) * speech_gain
        if emotion.dim() == 1:
            emotion = emotion[:, None].expand(B, T)
        if intensity.dim() == 1:
            intensity = intensity[:, None].expand(B, T)
        inten = intensity[..., None].float()
        # The emotion pathway may see prosody, but is trained to work without it (TTS prosody is neutral).
        pr_e = prosody if emo_prosody_keep is None else prosody * emo_prosody_keep[:, None, None]
        # ...and sometimes without the speech context, so emotion does not become sentence-specific.
        h_e = h if emo_ctx_keep is None else h * emo_ctx_keep[:, None, None]
        e = torch.cat([self.emo_emb(emotion) * inten, inten, pr_e, h_e], -1)
        g = self.emo_inp(e) + sinusoid(T, self.emo_inp.out_features, dev)[None]
        for blk in self.emo_blocks:
            g = blk(g, pad_mask)
        emo = self.emo_out(g) * inten.clamp(0, 1.5)
        logits = self.tone + speech + emo
        acts = torch.sigmoid(logits) if self.output == "sigmoid" else logits
        head = self.head_out(torch.cat([h, g], -1)) * 0.3
        if return_parts:
            return acts, head, {"speech": speech, "emotion": emo}
        return acts, head
