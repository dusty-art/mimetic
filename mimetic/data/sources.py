"""Dataset registry: where each corpus comes from, its licence, and how to label its clips.

Every processed clip records `dataset` so a commercially clean model can be
retrained later from the permissive subset (GRID: CC BY 4.0, CREMA-D: ODbL).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

LICENSES = {
    "grid": ("CC-BY-4.0", True, "https://doi.org/10.5281/zenodo.3625687"),
    "ravdess": ("CC-BY-NC-SA-4.0", False, "https://doi.org/10.5281/zenodo.1188976"),
    "cremad": ("ODbL-1.0 / DbCL-1.0", True, "https://github.com/CheyneyComputerScience/CREMA-D"),
}

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "contempt", "calm"]


@dataclass
class Unit:
    dataset: str
    name: str
    url: str
    archive: str


def grid_units():
    for n in range(1, 35):
        if n == 21:  # no video for talker 21
            continue
        yield Unit("grid", f"s{n}", f"https://zenodo.org/records/3625687/files/s{n}.zip?download=1", f"s{n}.zip")


def ravdess_units():
    for n in range(1, 25):
        yield Unit("ravdess", f"actor{n:02d}",
                   f"https://zenodo.org/records/1188976/files/Video_Speech_Actor_{n:02d}.zip?download=1",
                   f"Video_Speech_Actor_{n:02d}.zip")


def cremad_units():
    yield Unit("cremad", "all", "https://huggingface.co/datasets/ameerazam08/CREMA_DATA/resolve/main/crema_video.zip",
               "crema_video.zip")


# --------------------------------------------------------------------------- labels
RAVDESS_EMO = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgusted", 8: "surprised"}
RAVDESS_TEXT = {1: "Kids are talking by the door", 2: "Dogs are sitting by the door"}

CREMA_TEXT = {
    "IEO": "It's eleven o'clock", "TIE": "That is exactly what happened", "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about", "TAI": "The airplane is almost full", "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock", "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket", "ITS": "I think I've seen this before", "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}
CREMA_EMO = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
CREMA_LVL = {"LO": 1, "MD": 2, "HI": 3, "XX": 2}


def grid_alignment_dir(stems: set[str], root: Path) -> Path | None:
    """GRID's alignments.zip labels some speaker folders differently from the video archives.

    Each talker read a unique set of 1000 sentence codes, so the right folder is
    the one whose file names overlap the archive's videos.
    """
    best, best_n = None, 0
    for d in sorted((root / "alignments").iterdir()):
        if not d.is_dir():
            continue
        n = len(stems & {p.stem for p in d.glob("*.align")})
        if n > best_n:
            best, best_n = d, n
    return best if best_n > 0.5 * len(stems) else None


def clip_meta(dataset: str, video: Path, root: Path, align_dir: Path | None = None) -> dict | None:
    """Labels for one video, or None if the clip should be skipped."""
    stem = video.stem
    if dataset == "ravdess":
        parts = [int(p) for p in stem.split("-")]
        if len(parts) != 7 or parts[0] != 1 or parts[1] != 1:   # full audio-visual speech only
            return None
        _, _, emo, inten, stmt, rep, actor = parts
        return {"dataset": "ravdess", "speaker": f"ravdess_{actor:02d}", "gender": "m" if actor % 2 else "f",
                "emotion": RAVDESS_EMO[emo], "intensity": inten, "text": RAVDESS_TEXT[stmt], "clip": stem}
    if dataset == "cremad":
        m = re.match(r"(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})", stem)
        if not m:
            return None
        actor, sent, emo, lvl = m.groups()
        return {"dataset": "cremad", "speaker": f"cremad_{actor}", "emotion": CREMA_EMO[emo],
                "intensity": CREMA_LVL[lvl], "text": CREMA_TEXT[sent], "clip": stem}
    if dataset == "grid":
        spk = video.parent.name
        align = (align_dir or (root / "alignments" / spk)) / f"{stem}.align"
        words = []
        if align.exists():
            for line in open(align):
                a, b, w = line.split()
                words.append((int(a) / 25000.0, int(b) / 25000.0, w))
        text = " ".join(w for _, _, w in words if w not in ("sil", "sp"))
        if not text:
            return None
        return {"dataset": "grid", "speaker": f"grid_{spk}", "emotion": "neutral", "intensity": 1,
                "text": text, "clip": stem,
                "word_times": [[a, b] for a, b, w in words if w not in ("sil", "sp")],
                "words": [w for _, _, w in words if w not in ("sil", "sp")]}
    raise ValueError(dataset)


VIDEO_EXT = {".mp4", ".mpg", ".flv", ".avi", ".mov"}
