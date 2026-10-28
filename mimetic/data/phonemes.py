"""Unified English phoneme inventory with articulatory targets.

Two front ends map into it: espeak IPA tokens (used to force-align real
recordings) and misaki/Kokoro phoneme characters (used by the text-to-speech
path). The articulatory table is what makes lip-sync physically checkable:
each phoneme states what the lips and jaw must achieve, independent of any
dataset's tracking quality.
"""
from __future__ import annotations

import numpy as np

VOWELS = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
CONSONANTS = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH",
              "T", "TH", "V", "W", "Y", "Z", "ZH"]
PHONES = ["SIL"] + VOWELS + CONSONANTS
PID = {p: i for i, p in enumerate(PHONES)}
N_PHONES = len(PHONES)

# Articulatory targets per phoneme (values in [0, 1]):
#   close : lips must seal (bilabial stops / nasal)
#   labdent: lower lip to upper teeth
#   round : lip rounding / protrusion
#   spread: lip spreading (corners back)
#   open  : jaw opening
#   dental: tongue tip between the teeth
#   crit  : how critical the lip target is (bilabial 1.0, vowels lower: coarticulation can bend them)
_A = {
    #        close labdent round spread open dental crit
    "SIL": (0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0),
    "AA":  (0.0, 0.0, 0.0, 0.1, 0.90, 0.0, 0.5),
    "AE":  (0.0, 0.0, 0.0, 0.4, 0.75, 0.0, 0.5),
    "AH":  (0.0, 0.0, 0.0, 0.1, 0.55, 0.0, 0.4),
    "AO":  (0.0, 0.0, 0.6, 0.0, 0.75, 0.0, 0.5),
    "AW":  (0.0, 0.0, 0.5, 0.0, 0.75, 0.0, 0.5),
    "AY":  (0.0, 0.0, 0.0, 0.3, 0.75, 0.0, 0.5),
    "EH":  (0.0, 0.0, 0.0, 0.4, 0.55, 0.0, 0.4),
    "ER":  (0.0, 0.0, 0.5, 0.0, 0.35, 0.0, 0.4),
    "EY":  (0.0, 0.0, 0.0, 0.5, 0.45, 0.0, 0.4),
    "IH":  (0.0, 0.0, 0.0, 0.5, 0.30, 0.0, 0.4),
    "IY":  (0.0, 0.0, 0.0, 0.8, 0.20, 0.0, 0.5),
    "OW":  (0.0, 0.0, 0.8, 0.0, 0.50, 0.0, 0.6),
    "OY":  (0.0, 0.0, 0.7, 0.0, 0.55, 0.0, 0.5),
    "UH":  (0.0, 0.0, 0.6, 0.0, 0.30, 0.0, 0.5),
    "UW":  (0.0, 0.0, 1.0, 0.0, 0.20, 0.0, 0.8),
    "B":   (1.0, 0.0, 0.0, 0.0, 0.10, 0.0, 1.0),
    "P":   (1.0, 0.0, 0.0, 0.0, 0.10, 0.0, 1.0),
    "M":   (1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 1.0),
    "F":   (0.0, 1.0, 0.0, 0.0, 0.10, 0.0, 0.9),
    "V":   (0.0, 1.0, 0.0, 0.0, 0.10, 0.0, 0.9),
    "TH":  (0.0, 0.0, 0.0, 0.1, 0.20, 1.0, 0.7),
    "DH":  (0.0, 0.0, 0.0, 0.1, 0.20, 1.0, 0.6),
    "W":   (0.0, 0.0, 1.0, 0.0, 0.15, 0.0, 0.9),
    "SH":  (0.0, 0.0, 0.7, 0.0, 0.15, 0.0, 0.7),
    "ZH":  (0.0, 0.0, 0.7, 0.0, 0.15, 0.0, 0.7),
    "CH":  (0.0, 0.0, 0.6, 0.0, 0.15, 0.0, 0.7),
    "JH":  (0.0, 0.0, 0.6, 0.0, 0.15, 0.0, 0.7),
    "R":   (0.0, 0.0, 0.4, 0.0, 0.20, 0.0, 0.5),
    "S":   (0.0, 0.0, 0.0, 0.4, 0.10, 0.0, 0.5),
    "Z":   (0.0, 0.0, 0.0, 0.4, 0.10, 0.0, 0.5),
    "T":   (0.0, 0.0, 0.0, 0.2, 0.15, 0.0, 0.3),
    "D":   (0.0, 0.0, 0.0, 0.2, 0.15, 0.0, 0.3),
    "N":   (0.0, 0.0, 0.0, 0.2, 0.15, 0.0, 0.3),
    "L":   (0.0, 0.0, 0.0, 0.2, 0.25, 0.0, 0.3),
    "K":   (0.0, 0.0, 0.0, 0.1, 0.20, 0.0, 0.2),
    "G":   (0.0, 0.0, 0.0, 0.1, 0.20, 0.0, 0.2),
    "NG":  (0.0, 0.0, 0.0, 0.1, 0.20, 0.0, 0.2),
    "HH":  (0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.1),
    "Y":   (0.0, 0.0, 0.0, 0.6, 0.15, 0.0, 0.4),
}
ARTIC_NAMES = ["close", "labdent", "round", "spread", "open", "dental", "crit"]
ARTIC = np.array([_A[p] for p in PHONES], np.float32)           # (N_PHONES, 7)

# espeak IPA token -> unified phones
_ESPEAK = {
    "p": ["P"], "b": ["B"], "t": ["T"], "d": ["D"], "k": ["K"], "ɡ": ["G"], "g": ["G"], "m": ["M"], "n": ["N"],
    "ŋ": ["NG"], "f": ["F"], "v": ["V"], "θ": ["TH"], "ð": ["DH"], "s": ["S"], "z": ["Z"], "ʃ": ["SH"],
    "ʒ": ["ZH"], "h": ["HH"], "tʃ": ["CH"], "dʒ": ["JH"], "l": ["L"], "ɹ": ["R"], "r": ["R"], "w": ["W"],
    "j": ["Y"], "ɾ": ["D"], "ʔ": ["T"], "n̩": ["N"], "l̩": ["L"], "ɫ": ["L"],
    "ə": ["AH"], "ɐ": ["AH"], "ʌ": ["AH"], "ᵻ": ["IH"], "ɪ": ["IH"], "i": ["IY"], "iː": ["IY"], "ɛ": ["EH"],
    "e": ["EH"], "eɪ": ["EY"], "æ": ["AE"], "a": ["AE"], "aɪ": ["AY"], "aʊ": ["AW"], "ɑ": ["AA"], "ɑː": ["AA"],
    "ɒ": ["AA"], "ɔ": ["AO"], "ɔː": ["AO"], "oʊ": ["OW"], "o": ["OW"], "oː": ["OW"], "əʊ": ["OW"],
    "ɔɪ": ["OY"], "ʊ": ["UH"], "u": ["UW"], "uː": ["UW"], "ɜ": ["ER"], "ɜː": ["ER"], "ɚ": ["ER"],
    "ɑːɹ": ["AA", "R"], "ɔːɹ": ["AO", "R"], "oːɹ": ["AO", "R"], "ɪɹ": ["IH", "R"], "ɛɹ": ["EH", "R"],
    "ʊɹ": ["UH", "R"], "əl": ["AH", "L"], "aɪə": ["AY", "AH"], "aɪɚ": ["AY", "ER"], "iə": ["IY", "AH"],
    "eə": ["EH", "AH"], "ʊə": ["UH", "AH"], "ju": ["Y", "UW"],
}


def espeak_to_unified(tok: str) -> list[str]:
    if tok in _ESPEAK:
        return _ESPEAK[tok]
    t = tok.replace("ː", "")
    if t in _ESPEAK:
        return _ESPEAK[t]
    return []


# misaki (Kokoro) phoneme characters -> unified phone
_MISAKI = {
    "b": "B", "d": "D", "f": "F", "h": "HH", "j": "Y", "k": "K", "l": "L", "m": "M", "n": "N", "p": "P",
    "s": "S", "t": "T", "v": "V", "w": "W", "z": "Z", "ɡ": "G", "g": "G", "ŋ": "NG", "ɹ": "R", "r": "R",
    "ʃ": "SH", "ʒ": "ZH", "ð": "DH", "θ": "TH", "ʧ": "CH", "ʤ": "JH", "ɾ": "D", "ʔ": "T",
    "ə": "AH", "ᵊ": "AH", "ɐ": "AH", "ʌ": "AH", "æ": "AE", "ɑ": "AA", "ɒ": "AA", "ɔ": "AO", "ɛ": "EH",
    "ɜ": "ER", "ɚ": "ER", "ɝ": "ER", "ɪ": "IH", "ᵻ": "IH", "i": "IY", "ʊ": "UH", "u": "UW",
    "A": "EY", "I": "AY", "O": "OW", "W": "AW", "Y": "OY", "Q": "OW", "a": "AA", "e": "EH", "o": "OW",
}
MISAKI_MODIFIERS = set("ˈˌː")


def misaki_to_unified(ch: str) -> str | None:
    if ch in _MISAKI:
        return _MISAKI[ch]
    if ch in MISAKI_MODIFIERS:
        return None          # stress/length: duration folds into the neighbouring phone
    return "SIL"            # spaces, punctuation


def frames_from_segments(segs, n_frames: int, fps: float) -> np.ndarray:
    """[(start_s, end_s, phone)] -> per-frame phone id (n_frames,), SIL where uncovered."""
    out = np.zeros(n_frames, np.int64)
    for a, b, p in segs:
        i0 = int(np.floor(a * fps + 0.5))
        i1 = max(i0 + 1, int(np.floor(b * fps + 0.5)))
        out[max(0, i0):min(n_frames, i1)] = PID[p]
    return out
