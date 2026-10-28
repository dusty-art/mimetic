"""Local web demo: type text (with emotion tags) or talk into the mic, watch the muscles drive the face.

    python -m mimetic.web.server   ->  http://127.0.0.1:8765
"""
from __future__ import annotations

import os
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mimetic.face import anatomy as A
from mimetic.face import canonical as C
from mimetic.face.render import loop_subdivision
from mimetic.runtime import outputs
from mimetic.runtime.engine import Engine

WEB = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Mimetic")
STATE = {}

VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael", "am_fenrir",
          "am_puck", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]


def engine() -> Engine:
    if "engine" not in STATE:
        STATE["engine"] = Engine.load(name=os.environ.get("MIMETIC_CONTROLLER", "controller"), device="cpu")
    return STATE["engine"]


def b64(a: np.ndarray, dtype=np.float32) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode()


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/mesh")
def mesh():
    eng = engine()
    face = C.load_canonical()
    S, faces = loop_subdivision(face.skin_faces, C.N_VERTS)
    S = S.tocoo()
    rest = eng.face.rest.cpu().numpy()
    from mimetic.face.sim import FaceSim
    sim = STATE.setdefault("sim", FaceSim(calibrate=False))
    muscles = []
    for m in sim.muscles:
        w = m.weights
        muscles.append({"name": m.spec.name, "kind": m.spec.kind, "facs": list(m.spec.facs),
                        "description": m.spec.description, "origin": m.origin.round(3).tolist(),
                        "jaw": m.spec.origin_frame == "jaw",
                        "ins_idx": np.nonzero(w > 0.05)[0].tolist(), "ins_w": w[w > 0.05].round(3).tolist()})
    for j in A.JAW:
        muscles.append({"name": j, "kind": "jaw", "facs": list(A.JAW_FACS[j]), "description": A.JAW_DESCRIPTIONS[j],
                        "origin": None, "jaw": True, "ins_idx": [], "ins_w": []})
    eyes = [{"c": c.round(3).tolist(), "r": float(r)} for c, r in sim.eyes]
    from mimetic.face.tongue import tongue_mesh
    tv, tf = tongue_mesh(np.zeros(5))
    lip = np.zeros(C.N_VERTS)
    ym = A.mouth_line_y(sim.face, rest[:, 0])
    lip[(np.abs(rest[:, 0]) < 2.7) & (rest[:, 1] < ym + 0.75) & (rest[:, 1] > ym - 1.0)] = 1
    return JSONResponse({
        "rest": b64(rest), "n": C.N_VERTS, "faces": b64(faces, np.int32), "n_faces": len(faces),
        "sub_rows": b64(S.row, np.int32), "sub_cols": b64(S.col, np.int32), "sub_vals": b64(S.data),
        "n_sub": int(S.shape[0]), "w_jaw": b64(sim.w_jaw.numpy()), "lip": b64(S @ lip),
        "muscles": muscles, "eyes": eyes, "names": A.MUSCLE_NAMES, "voices": VOICES,
        "arkit": outputs.ARKIT, "tongue_rest": b64(tv), "tongue_faces": b64(tf, np.int32), "n_tongue": len(tv)})


class SayReq(BaseModel):
    text: str
    voice: str = "af_heart"
    style: str | None = None
    gain: float = 1.0
    expressiveness: float = 1.0
    fps: float = 60.0
    seed: int = 0


def wav_bytes(audio, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.post("/api/say")
def say(req: SayReq):
    eng = engine()
    anim = eng.animate_text(req.text, voice=req.voice, style=req.style, fps=req.fps, speech_gain=req.gain,
                            seed=req.seed, emotion_gain=req.expressiveness)
    STATE["last"] = anim
    v = anim.vertices
    from mimetic.face.tongue import tongue_mesh
    tongue = np.stack([tongue_mesh(p)[0] for p in anim.tongue]) if anim.tongue is not None else None
    return JSONResponse({
        "tongue": b64(tongue.astype(np.float16), np.float16) if tongue is not None else None,
        "fps": anim.fps, "n": anim.n, "acts": b64(anim.acts), "verts": b64(v.astype(np.float16), np.float16),
        "head": b64(anim.head), "gaze": b64(anim.gaze), "arkit": b64(anim.arkit),
        "audio": base64.b64encode(wav_bytes(anim.audio, anim.sr)).decode(),
        "phonemes": [[round(a, 3), round(b, 3), p] for a, b, p in anim.segments],
        "emotions": [[round(a, 3), round(b, 3), e, i] for a, b, e, i in anim.emotion_spans]})


@app.get("/api/export/{fmt}")
def export(fmt: str):
    anim = STATE.get("last")
    if anim is None:
        return JSONResponse({"error": "nothing generated yet"}, status_code=404)
    tmp = Path("/tmp") / f"mimetic_export.{ 'csv' if fmt == 'livelink' else 'json'}"
    if fmt == "livelink":
        outputs.write_livelink_csv(tmp, anim)
        return FileResponse(tmp, filename="mimetic_livelink.csv", media_type="text/csv")
    outputs.write_json(tmp, anim, include_vertices=(fmt == "json_full"))
    return FileResponse(tmp, filename="mimetic_animation.json", media_type="application/json")


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    """Browser streams 16 kHz mono float32 PCM; we answer with vertices + activations per 10 ms hop."""
    await ws.accept()
    from mimetic.runtime.live import AudioDriver
    eng = engine()
    drv = AudioDriver.load("cpu")
    emotion = ("auto", 1.0)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("text"):
                cfg = json.loads(msg["text"])
                if "emotion" in cfg:
                    emotion = (cfg["emotion"], float(cfg.get("intensity", 0.8)))
                continue
            pcm = np.frombuffer(msg["bytes"], np.float32)
            frames = drv.push(pcm, emotion)
            if not frames:
                continue
            a = np.stack(frames)[-1:]                       # render the newest frame only
            with torch.no_grad():
                v = eng.face(torch.as_tensor(a, dtype=torch.float32)).numpy()
            emo = getattr(drv, "last_emotion", None)
            from mimetic.face.tongue import tongue_mesh
            tg = tongue_mesh(drv.last_tongue[-1])[0] if getattr(drv, "last_tongue", None) else None
            await ws.send_text(json.dumps({"acts": a[0].round(3).tolist(), "verts": b64(v[0].astype(np.float16), np.float16),
                                           "tongue": b64(tg.astype(np.float16), np.float16) if tg is not None else None,
                                           "vad": getattr(drv, "last_vad", 0.0),
                                           "emotion": None if emo is None else emo.round(3).tolist()}))
    except WebSocketDisconnect:
        pass


app.mount("/static", StaticFiles(directory=WEB), name="static")


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
