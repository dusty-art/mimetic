"""Resumable background pipeline: download archive -> extract -> track every clip -> delete raw video.

Usage: python scripts/fetch_and_track.py [--workers 7] [--order ravdess:4,cremad,grid:4,ravdess,grid]

Each unit (one speaker's archive) is marked done in data/processed/_done/. Raw
archives and videos are deleted as soon as a unit is processed, so peak disk use
is about two archives.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mimetic.data import sources  # noqa: E402
from mimetic.paths import PROCESSED, RAW  # noqa: E402

LOG = PROCESSED / "_log.txt"
DONE = PROCESSED / "_done"
DISK_FLOOR_GB = 12.0   # stop if free disk drops below this (keeps the user's machine healthy)


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def free_gb():
    return shutil.disk_usage(str(PROCESSED)).free / 1e9


def download(unit: sources.Unit) -> Path:
    d = RAW / unit.dataset
    d.mkdir(parents=True, exist_ok=True)
    out = d / unit.archive
    if out.exists() and not (d / (unit.archive + ".aria2")).exists():
        return out
    for attempt in range(5):
        r = subprocess.run(["aria2c", "-q", "-x", "8", "-s", "8", "-c", "--max-tries=5", "--retry-wait=5",
                            "-d", str(d), "-o", unit.archive, unit.url])
        if r.returncode == 0:
            return out
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed: {unit.url}")


def _work(args):
    video, out, meta = args
    from mimetic.data import track
    try:
        return video, track.process_clip(video, out, meta)
    except Exception as e:  # keep going on corrupt clips
        return video, f"error: {e}"


def process_unit(unit: sources.Unit, archive: Path, pool):
    tmp = RAW / unit.dataset / f"_x_{unit.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if Path(m).suffix.lower() in sources.VIDEO_EXT and "__MACOSX" not in m]
        z.extractall(tmp, members)
    root = RAW / unit.dataset
    jobs = []
    for v in sorted(tmp.rglob("*")):
        if v.suffix.lower() not in sources.VIDEO_EXT:
            continue
        meta = sources.clip_meta(unit.dataset, v, root)
        if meta is None:
            continue
        out = PROCESSED / unit.dataset / meta["speaker"] / f"{meta['clip']}.npz"
        jobs.append((str(v), str(out), meta))
    t0 = time.time()
    ok = bad = 0
    futs = [pool.submit(_work, j) for j in jobs]
    for f in as_completed(futs):
        _, res = f.result()
        if res is True:
            ok += 1
        else:
            bad += 1
    shutil.rmtree(tmp)
    archive.unlink(missing_ok=True)
    log(f"{unit.dataset}/{unit.name}: {ok} clips ok, {bad} skipped in {time.time() - t0:.0f}s, free {free_gb():.1f} GB")


def parse_order(spec):
    gens = {"grid": list(sources.grid_units()), "ravdess": list(sources.ravdess_units()),
            "cremad": list(sources.cremad_units())}
    seen, units = set(), []
    for part in spec.split(","):
        name, _, n = part.partition(":")
        lst = gens[name][: int(n)] if n else gens[name]
        for u in lst:
            if (u.dataset, u.name) not in seen:
                seen.add((u.dataset, u.name))
                units.append(u)
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--order", default="ravdess:4,cremad,grid:4,ravdess,grid")
    args = ap.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    DONE.mkdir(parents=True, exist_ok=True)
    units = [u for u in parse_order(args.order) if not (DONE / f"{u.dataset}_{u.name}").exists()]
    log(f"starting: {len(units)} units pending, {args.workers} workers, free {free_gb():.1f} GB")
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"))
    # Prefetch the next archive while the current one is processed.
    prefetch = {}

    def fetch(i):
        try:
            prefetch[i] = download(units[i])
        except Exception as e:
            prefetch[i] = e

    th = threading.Thread(target=fetch, args=(0,)) if units else None
    if th:
        th.start()
    for i, u in enumerate(units):
        th.join()
        arch = prefetch.pop(i)
        if i + 1 < len(units):
            th = threading.Thread(target=fetch, args=(i + 1,))
            th.start()
        if isinstance(arch, Exception):
            log(f"{u.dataset}/{u.name}: download failed: {arch}")
            continue
        if free_gb() < DISK_FLOOR_GB:
            log(f"stopping: free disk {free_gb():.1f} GB below floor")
            break
        try:
            process_unit(u, arch, pool)
            (DONE / f"{u.dataset}_{u.name}").touch()
        except Exception:
            log(f"{u.dataset}/{u.name}: failed\n{traceback.format_exc()}")
    pool.shutdown()
    log("finished")


if __name__ == "__main__":
    os.environ.setdefault("GLOG_minloglevel", "2")
    main()
