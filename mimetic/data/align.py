"""Phoneme-level forced alignment of known transcripts.

espeak-ng phonemises the transcript, a wav2vec2 phoneme recogniser
(facebook/wav2vec2-lv-60-espeak-cv-ft, Apache-2.0) produces frame posteriors
at 50 Hz, and a CTC Viterbi pass places every phoneme in time.
"""
from __future__ import annotations

import numpy as np
import torch

from mimetic.data import phonemes as P

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"


class Aligner:
    def __init__(self, device: str | None = None):
        import mimetic  # noqa: F401  (sets HF_HOME)
        from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
        import json
        from huggingface_hub import hf_hub_download
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(self.device).eval()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        self.vocab = json.load(open(hf_hub_download(MODEL_ID, "vocab.json")))
        self.blank = self.vocab["<pad>"]
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator
        self.backend = EspeakBackend("en-us", preserve_punctuation=False, with_stress=False)
        self.sep = Separator(phone=" ", word=" | ", syllable="")

    def phonemize(self, text: str):
        """-> list of (espeak token, word index)."""
        out = self.backend.phonemize([text], separator=self.sep, strip=True)[0]
        toks, w = [], 0
        for t in out.split(" "):
            if t == "|":
                w += 1
            elif t:
                toks.append((t, w))
        return toks

    @torch.no_grad()
    def posteriors(self, audio16k: np.ndarray) -> np.ndarray:
        x = self.fe(audio16k, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        logits = self.model(x).logits[0].float()
        return torch.log_softmax(logits, -1).cpu().numpy()      # (T, V) at 50 Hz

    def tokenize(self, toks):
        ids, keep = [], []
        for t, w in toks:
            if t in self.vocab:
                ids.append(self.vocab[t]); keep.append((t, w))
            else:  # split unknown multi-char tokens into known pieces
                for ch in t:
                    if ch in self.vocab:
                        ids.append(self.vocab[ch]); keep.append((ch, w))
        return ids, keep

    def align(self, audio16k: np.ndarray, text: str):
        """-> list of (start_s, end_s, unified_phone, word_idx), sorted, SIL-free."""
        toks = self.phonemize(text)
        ids, keep = self.tokenize(toks)
        if not ids:
            return []
        lp = self.posteriors(audio16k)
        spans = ctc_viterbi(lp, ids, self.blank)
        if spans is None:
            return []
        return spikes_to_segments(spans, keep)


def spikes_to_segments(spans, keep, hop=0.02, pause=0.30):
    """CTC emits each phone as a short spike; grow spikes into contiguous phone intervals.

    Boundaries sit midway between neighbouring spikes; gaps longer than `pause`
    become silence, with the phones on either side keeping a natural tail.
    """
    raw = []
    for (s, e), (tok, w) in zip(spans, keep):
        ph = P.espeak_to_unified(tok)
        if ph:
            raw.append((s * hop, (e + 1) * hop, ph, w))
    if not raw:
        return []
    starts = [r[0] for r in raw]
    ends = [r[1] for r in raw]
    bounds_lo, bounds_hi = [], []
    for k in range(len(raw)):
        if k == 0 or starts[k] - ends[k - 1] > pause:
            lo = starts[k] - 0.05
        else:
            lo = (ends[k - 1] + starts[k]) / 2
        if k == len(raw) - 1 or starts[k + 1] - ends[k] > pause:
            hi = ends[k] + 0.08
        else:
            hi = (ends[k] + starts[k + 1]) / 2
        bounds_lo.append(max(0.0, lo)); bounds_hi.append(hi)
    segs = []
    for (a, b), (_, _, ph, w) in zip(zip(bounds_lo, bounds_hi), raw):
        step = (b - a) / len(ph)
        for j, p in enumerate(ph):
            segs.append((round(a + j * step, 4), round(a + (j + 1) * step, 4), p, w))
    return segs


def ctc_viterbi(logp: np.ndarray, ids: list[int], blank: int):
    """Forced CTC alignment. Returns [(first_frame, last_frame)] per token, or None if impossible."""
    T = logp.shape[0]
    L = len(ids)
    S = 2 * L + 1
    ext = np.full(S, blank)
    ext[1::2] = ids
    NEG = -1e30
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), np.int8)
    dp[0, 0] = logp[0, blank]
    if S > 1:
        dp[0, 1] = logp[0, ext[1]]
    skip_ok = np.zeros(S, bool)
    skip_ok[3::2] = ext[3::2] != ext[1:-2:2]
    for t in range(1, T):
        prev = dp[t - 1]
        c0 = prev
        c1 = np.concatenate([[NEG], prev[:-1]])
        c2 = np.concatenate([[NEG, NEG], prev[:-2]])
        c2 = np.where(skip_ok, c2, NEG)
        stack = np.stack([c0, c1, c2])
        arg = stack.argmax(0)
        dp[t] = stack[arg, np.arange(S)] + logp[t, ext]
        bp[t] = arg
    end = S - 1 if dp[-1, S - 1] >= (dp[-1, S - 2] if S > 1 else NEG) else S - 2
    if dp[-1, end] <= NEG / 2:
        return None
    path = np.zeros(T, np.int64)
    s = end
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    spans = [[None, None] for _ in range(L)]
    for t, s in enumerate(path):
        if s % 2 == 1:
            k = s // 2
            if spans[k][0] is None:
                spans[k][0] = t
            spans[k][1] = t
    # Tokens that never got a frame (very short) borrow the boundary of their neighbour.
    for k in range(L):
        if spans[k][0] is None:
            prev_end = spans[k - 1][1] if k > 0 and spans[k - 1][1] is not None else 0
            spans[k] = [prev_end, prev_end]
    return [tuple(s) for s in spans]
