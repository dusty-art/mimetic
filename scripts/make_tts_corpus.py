"""Synthetic speech corpus for the audio model: public-domain sentences x 28 Kokoro voices.

Every utterance comes with exact phoneme timing (from the TTS duration
predictor), which the controller turns into teacher muscle activations later.
Output: data/tts/{id}.flac + data/tts/index.jsonl
"""
from __future__ import annotations

import io
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from mimetic.paths import DATA  # noqa: E402

BOOKS = {  # Project Gutenberg, public domain in the US
    "pride_prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "sherlock": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "alice": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "earnest": "https://www.gutenberg.org/cache/epub/844/pg844.txt",
}
VOICES_A = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
            "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
            "am_michael", "am_onyx", "am_puck"]
VOICES_B = ["bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
EMOS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "contempt", "calm"]
OUT = DATA / "tts"


def sentences():
    txt_dir = DATA / "text"
    txt_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, url in BOOKS.items():
        p = txt_dir / f"{name}.txt"
        if not p.exists():
            p.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        t = p.read_text(errors="ignore")
        a, b = t.find("*** START"), t.find("*** END")
        t = t[t.find("\n", a) + 1: b] if a >= 0 and b > a else t
        t = re.sub(r"\s+", " ", t.replace("_", ""))
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", t):
            s = s.strip(" \"'")
            w = s.split()
            if 4 <= len(w) <= 22 and not re.search(r"[0-9\[\]{}*#|]|CHAPTER", s):
                out.append(s)
    return out


def main(n=4000, seed=0):
    from mimetic.runtime.tts import TextToSpeech
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sents = sentences()
    rng.shuffle(sents)
    print(f"{len(sents)} candidate sentences", flush=True)
    idx_path = OUT / "index.jsonl"
    done = set()
    if idx_path.exists():
        done = {json.loads(l)["id"] for l in idx_path.open()}
    tts = {"a": TextToSpeech(lang="a"), "b": TextToSpeech(lang="b")}
    t0 = time.time()
    total = 0.0
    with idx_path.open("a") as f:
        for i, s in enumerate(sents[:n]):
            uid = f"u{i:05d}"
            if uid in done:
                continue
            voice = rng.choice(VOICES_A + VOICES_B)
            lang = "b" if voice[0] == "b" else "a"
            emo = rng.choice(EMOS)
            inten = 0.0 if emo == "neutral" else round(rng.uniform(0.4, 1.0), 2)
            rate = rng.uniform(0.9, 1.12)
            try:
                u = tts[lang].synthesize(f"[{emo}:{inten}] {s}" if emo != "neutral" else s, voice=voice, rate=rate)
            except Exception as e:
                print("skip", uid, e, flush=True)
                continue
            if len(u.audio) == 0:
                continue
            a16 = np.interp(np.arange(int(len(u.audio) * 16000 / 24000)) / 16000,
                            np.arange(len(u.audio)) / 24000, u.audio).astype(np.float32)
            sf.write(OUT / f"{uid}.flac", a16, 16000)
            total += len(a16) / 16000
            f.write(json.dumps({"id": uid, "text": s, "voice": voice, "emotion": emo, "intensity": inten,
                                "segments": [[round(a, 4), round(b, 4), p] for a, b, p in u.segments],
                                "dur": len(a16) / 16000}) + "\n")
            f.flush()
            if (i + 1) % 200 == 0:
                print(f"{i + 1} utterances, {total / 60:.1f} min audio, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {total / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
