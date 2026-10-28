"""Force-align phonemes for every processed clip (writes {clip}.phn.json next to each .npz).

Runs alongside the tracking pipeline: it keeps polling for new clips until the
tracker logs "finished" and nothing is left to align.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from mimetic.data import track  # noqa: E402
from mimetic.data.align import Aligner  # noqa: E402
from mimetic.paths import PROCESSED  # noqa: E402


def pending():
    out = []
    for f in PROCESSED.glob("*/*/*.npz"):
        if f.name.endswith(".tmp.npz"):
            continue
        if not f.with_suffix(".phn.json").exists():
            out.append(f)
    return sorted(out)


def main():
    al = Aligner()
    n_done, t0 = 0, time.time()
    while True:
        todo = pending()
        if not todo:
            log = (PROCESSED / "_log.txt").read_text() if (PROCESSED / "_log.txt").exists() else ""
            if "finished" in log.splitlines()[-1:][0] if log else False:
                break
            time.sleep(30)
            continue
        for f in todo:
            try:
                z = np.load(f)
                text = str(z["text"])
                d = track.load_clip(f)
                segs = al.align(d["audio"], text)
                rec = {"text": text, "segments": [[a, b, p, int(w)] for a, b, p, w in segs]}
            except Exception as e:
                rec = {"error": str(e), "segments": []}
            f.with_suffix(".phn.json").write_text(json.dumps(rec))
            n_done += 1
            if n_done % 500 == 0:
                print(f"{time.strftime('%H:%M:%S')} aligned {n_done} clips ({n_done / (time.time() - t0):.1f}/s)", flush=True)
    print(f"done: {n_done} clips", flush=True)


if __name__ == "__main__":
    main()
