"""The CREMA-D video mirror has no audio track; attach the matching 16 kHz WAV to each tracked clip."""
import io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, soundfile as sf
from mimetic.paths import PROCESSED, RAW

wavs = RAW / "cremad_audio" / "audios"
n = 0
for f in sorted((PROCESSED / "cremad").glob("*/*.npz")):
    z = dict(np.load(f))
    try:
        sf.read(io.BytesIO(z["audio_flac"].tobytes()))
        continue                     # already has audio
    except Exception:
        pass
    w = wavs / f"{str(z['clip'])}.wav"
    if not w.exists():
        continue
    a, sr = sf.read(w, dtype="int16")
    assert sr == 16000
    buf = io.BytesIO(); sf.write(buf, a, 16000, format="FLAC")
    z["audio_flac"] = np.frombuffer(buf.getvalue(), np.uint8)
    tmp = f.parent / (f.stem + ".tmp.npz")
    np.savez_compressed(tmp, **z); tmp.rename(f)
    f.with_suffix(".phn.json").unlink(missing_ok=True)
    n += 1
print("attached audio to", n, "clips")
