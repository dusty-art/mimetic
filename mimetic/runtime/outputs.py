"""Outputs for avatars and tools.

* ARKit 52 blendshape weights (iPhone / "Perfect Sync" VRM / MetaHuman / most 3D avatars),
  learned from real faces where both muscle fits and tracked blendshapes exist.
* VTube Studio / Live2D parameters, measured geometrically on the simulated face.
* Live Link Face CSV (Unreal, Blender importers), JSON, VMC protocol (OSC) and the
  VTube Studio plugin API for live streaming.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn

from mimetic.data.motion import BS_NAMES
from mimetic.face import anatomy as A
from mimetic.model.articulation import measures

ARKIT = BS_NAMES[1:]          # 52 names, MediaPipe order (ARKit-compatible)
ARKIT_CANONICAL_ORDER = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeSquintLeft",
    "eyeWideLeft", "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
    "eyeSquintRight", "eyeWideRight", "jawForward", "jawLeft", "jawRight", "jawOpen", "mouthClose",
    "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight", "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight", "tongueOut"]
GAZE_KEYS = [n for n in ARKIT if n.startswith("eyeLook")]


class ArkitMap(nn.Module):
    """Muscle activations -> ARKit blendshape weights (gaze channels come from the gaze track)."""

    def __init__(self, n_act=A.N_ACT, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_act, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, len(ARKIT)))

    def forward(self, a):
        return torch.sigmoid(self.net(a))

    @classmethod
    def load(cls, path, device="cpu"):
        m = cls()
        ck = torch.load(path, map_location="cpu")
        m.load_state_dict(ck["state"])
        m.r2 = ck.get("r2", {})
        return m.to(device).eval()


def facs_arkit(acts):
    """ARKit weights straight from muscle activations via FACS (muscle -> action unit -> blendshape).

    Used for channels the learned map cannot predict reliably (rare in tracked video).
    Sides: Mimetic "_L" = the person's left = ARKit "Left".
    """
    nm = A.MUSCLE_NAMES
    g = lambda n: acts[:, nm.index(n)]
    out = {}
    for s, S in (("L", "Left"), ("R", "Right")):
        out[f"eyeBlink{S}"] = g(f"orbicularis_oculi_palpebral_{s}")
        out[f"eyeSquint{S}"] = 0.7 * g(f"orbicularis_oculi_orbital_{s}")
        out[f"eyeWide{S}"] = g(f"levator_palpebrae_{s}")
        out[f"cheekSquint{S}"] = g(f"orbicularis_oculi_orbital_{s}")
        out[f"noseSneer{S}"] = g(f"levator_labii_alaeque_nasi_{s}")
        out[f"mouthSmile{S}"] = g(f"zygomaticus_major_{s}")
        out[f"mouthFrown{S}"] = g(f"depressor_anguli_oris_{s}")
        out[f"mouthStretch{S}"] = np.clip(g(f"risorius_{s}") + 0.5 * g(f"platysma_{s}"), 0, 1)
        out[f"mouthDimple{S}"] = g(f"buccinator_{s}")
        out[f"mouthPress{S}"] = 0.5 * (g(f"orbicularis_oris_marginal_upper_{s}") + g(f"orbicularis_oris_marginal_lower_{s}"))
        out[f"mouthUpperUp{S}"] = g(f"levator_labii_superioris_{s}")
        out[f"mouthLowerDown{S}"] = g(f"depressor_labii_inferioris_{s}")
        out[f"browDown{S}"] = np.maximum(g(f"corrugator_{s}"), g(f"depressor_supercilii_{s}"))
        out[f"browOuterUp{S}"] = g(f"frontalis_lateral_{s}")
    out["browInnerUp"] = 0.5 * (g("frontalis_medial_L") + g("frontalis_medial_R"))
    out["mouthRollUpper"] = 0.5 * (g("orbicularis_oris_marginal_upper_L") + g("orbicularis_oris_marginal_upper_R"))
    out["mouthRollLower"] = 0.5 * (g("orbicularis_oris_marginal_lower_L") + g("orbicularis_oris_marginal_lower_R"))
    out["mouthShrugLower"] = 0.5 * (g("mentalis_L") + g("mentalis_R"))
    out["mouthShrugUpper"] = 0.25 * (g("mentalis_L") + g("mentalis_R"))
    pucker = 0.25 * sum(g(f"orbicularis_oris_peripheral_{u}_{s}") for u in ("upper", "lower") for s in "LR")
    out["mouthPucker"] = pucker
    out["mouthFunnel"] = np.clip(pucker * (0.5 + 2 * g("jaw_open")), 0, 1)
    out["jawOpen"] = np.clip(1.8 * g("jaw_open"), 0, 1)
    out["jawForward"] = g("jaw_protrude")
    out["jawLeft"] = g("jaw_lateral_L")
    out["jawRight"] = g("jaw_lateral_R")
    out["mouthLeft"] = np.clip(g("zygomaticus_major_L") + g("risorius_L") - g("zygomaticus_major_R") - g("risorius_R"), 0, 1) * 0.5
    out["mouthRight"] = np.clip(g("zygomaticus_major_R") + g("risorius_R") - g("zygomaticus_major_L") - g("risorius_L"), 0, 1) * 0.5
    out["mouthClose"] = np.clip(g("jaw_open") * 2 - 1.5 * out["mouthLowerDownLeft"] - 1.5 * out["mouthUpperUpLeft"], 0, 1) * \
        0.5 * (g("orbicularis_oris_marginal_upper_L") + g("orbicularis_oris_marginal_lower_L") > 0.2)
    out["cheekPuff"] = np.zeros(len(acts))
    return out


def gaze_to_arkit(gaze):
    """gaze (T,2) yaw (+ = subject's left) / pitch (+ = up) radians -> eyeLook* weights."""
    yaw, pitch = gaze[:, 0], gaze[:, 1]
    s = 1 / 0.45
    out = {}
    out["eyeLookUpLeft"] = out["eyeLookUpRight"] = np.clip(pitch * s, 0, 1)
    out["eyeLookDownLeft"] = out["eyeLookDownRight"] = np.clip(-pitch * s, 0, 1)
    out["eyeLookOutLeft"] = np.clip(yaw * s, 0, 1)
    out["eyeLookInRight"] = np.clip(yaw * s, 0, 1)
    out["eyeLookInLeft"] = np.clip(-yaw * s, 0, 1)
    out["eyeLookOutRight"] = np.clip(-yaw * s, 0, 1)
    return out


@torch.no_grad()
def arkit_from_acts(engine, acts, gaze=None, tongue=None):
    a = torch.as_tensor(acts, dtype=torch.float32, device=engine.device)
    facs = facs_arkit(np.asarray(acts, np.float32))
    bs = np.zeros((len(acts), len(ARKIT)), np.float32)
    for k, v in facs.items():
        bs[:, ARKIT.index(k)] = v
    if engine.arkit_map is not None:
        # Learned map where it predicts real faces well, FACS mapping elsewhere.
        learned = engine.arkit_map(a).cpu().numpy()
        r2 = getattr(engine.arkit_map, "r2", {})
        for j, n in enumerate(ARKIT):
            if r2.get(n, 0.0) >= 0.3:
                bs[:, j] = learned[:, j]
    if gaze is not None:
        for k, v in gaze_to_arkit(gaze).items():
            bs[:, ARKIT.index(k)] = v
    return np.clip(bs, 0, 1)


def tongue_out(tongue):
    """ARKit tongueOut from the tongue track: tip forward past the teeth (th / dh)."""
    return np.clip((np.asarray(tongue)[:, 1] - 0.15) / 0.35, 0, 1)


# ---------------------------------------------------------------- VTube Studio / Live2D
@torch.no_grad()
def vts_params(engine, acts, head, gaze):
    """Default VTube Studio tracking parameters, measured on the simulated face."""
    a = torch.as_tensor(acts, dtype=torch.float32, device=engine.device)
    x = engine.face(a)
    rest = engine.face.rest
    m = measures(x)
    r = measures(rest[None])
    nm = A.MUSCLE_NAMES
    opening = ((m["aperture"] - r["aperture"] * 0.3) / 1.2).clamp(0, 1)
    corner_up = ((x[:, 61, 1] + x[:, 291, 1]) / 2 - (rest[61, 1] + rest[291, 1]) / 2)
    smile = ((corner_up + 0.5 * (m["width"] - r["width"])) / 0.6 + 0.5).clamp(0, 1)
    def lid(up, lo, ru, rl):
        ap = (x[:, up, 1] - x[:, lo, 1]) / (rest[ru, 1] - rest[rl, 1])
        return ap.clamp(0, 1.2)
    eyeL, eyeR = lid(386, 374, 386, 374), lid(159, 145, 159, 145)
    browL = ((x[:, 334, 1] - rest[334, 1]) / 0.6).clamp(-1, 1) * 0.5 + 0.5
    browR = ((x[:, 105, 1] - rest[105, 1]) / 0.6).clamp(-1, 1) * 0.5 + 0.5
    deg = 180 / math.pi
    p = {"MouthOpen": opening, "MouthSmile": smile, "EyeOpenLeft": eyeL, "EyeOpenRight": eyeR,
         "BrowLeftY": browL, "BrowRightY": browR, "Brows": (browL + browR) / 2}
    out = {k: v.cpu().numpy() for k, v in p.items()}
    out["FaceAngleX"] = head[:, 1] * deg
    out["FaceAngleY"] = head[:, 0] * deg
    out["FaceAngleZ"] = head[:, 2] * deg
    out["EyeLeftX"] = out["EyeRightX"] = np.clip(gaze[:, 0] / 0.45, -1, 1)
    out["EyeLeftY"] = out["EyeRightY"] = np.clip(gaze[:, 1] / 0.45, -1, 1)
    out["MouthX"] = np.clip(((x[:, 291, 0] + x[:, 61, 0]) / 2).cpu().numpy() / 0.5, -1, 1)
    out["FaceAngry"] = np.clip(acts[:, nm.index("corrugator_R")] + acts[:, nm.index("corrugator_L")], 0, 1)
    return out


# ---------------------------------------------------------------- file exports
def write_livelink_csv(path, anim):
    """Live Link Face CSV (Unreal / Blender importers): 52 blendshapes + head + eye rotations."""
    bs = anim.arkit
    names = [n[0].upper() + n[1:] for n in ARKIT_CANONICAL_ORDER]
    cols = ["Timecode", "BlendShapeCount"] + names + ["HeadYaw", "HeadPitch", "HeadRoll", "LeftEyeYaw",
                                                       "LeftEyePitch", "LeftEyeRoll", "RightEyeYaw",
                                                       "RightEyePitch", "RightEyeRoll"]
    idx = {n: ARKIT.index(n) if n in ARKIT else None for n in ARKIT_CANONICAL_ORDER}
    tout = tongue_out(anim.tongue) if anim.tongue is not None else np.zeros(anim.n)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for t in range(anim.n):
            sec = t / anim.fps
            hh, mm = int(sec // 3600), int(sec // 60) % 60
            ss, fr = int(sec) % 60, int((sec % 1) * anim.fps)
            frac = (sec * anim.fps) % 1
            tc = f"{hh:02d}:{mm:02d}:{ss:02d}:{fr:02d}.{int(frac * 1000):03d}"
            vals = [bs[t, idx[n]] if idx[n] is not None else (tout[t] if n == "tongueOut" else 0.0)
                    for n in ARKIT_CANONICAL_ORDER]
            h = anim.head[t]
            g = anim.gaze[t]
            rot = [h[1], h[0], h[2], g[0], g[1], 0.0, g[0], g[1], 0.0]
            f.write(",".join([tc, "61"] + [f"{v:.4f}" for v in vals] + [f"{v:.4f}" for v in rot]) + "\n")


def write_json(path, anim, include_vertices=False):
    rec = {"fps": anim.fps, "n_frames": anim.n, "muscles": A.MUSCLE_NAMES,
           "activations": np.round(anim.acts, 4).tolist(), "arkit_names": ARKIT,
           "arkit": np.round(anim.arkit, 4).tolist(), "head_pitch_yaw_roll": np.round(anim.head, 4).tolist(),
           "gaze_yaw_pitch": np.round(anim.gaze, 4).tolist(),
           "tongue_names": ["tip_up", "tip_fwd", "front_up", "back_up", "low"],
           "tongue_cm": np.round(anim.tongue, 4).tolist() if anim.tongue is not None else None,
           "tongueOut": np.round(tongue_out(anim.tongue), 4).tolist() if anim.tongue is not None else None,
           "phonemes": [[round(a, 3), round(b, 3), p] for a, b, p in anim.segments],
           "emotions": [[round(a, 3), round(b, 3), e, round(i, 3)] for a, b, e, i in anim.emotion_spans]}
    if include_vertices:
        rec["vertices_cm"] = np.round(anim.vertices, 3).tolist()
    with open(path, "w") as f:
        json.dump(rec, f)


# ---------------------------------------------------------------- live senders
class VMCSender:
    """VMC protocol over OSC/UDP (VSeeFace, Warudo, VNyan, VSeeFace-compatible receivers)."""

    def __init__(self, host="127.0.0.1", port=39539):
        from pythonosc.udp_client import SimpleUDPClient
        self.c = SimpleUDPClient(host, port)

    def send(self, arkit_row, head_row, tongue_out_value=None):
        for n, v in zip(ARKIT, arkit_row):
            self.c.send_message("/VMC/Ext/Blend/Val", [n, float(v)])
        if tongue_out_value is not None:
            self.c.send_message("/VMC/Ext/Blend/Val", ["tongueOut", float(tongue_out_value)])
        self.c.send_message("/VMC/Ext/Blend/Apply", [])
        p, y, r = [float(v) for v in head_row]
        cy, sy, cp, sp, cr, sr = math.cos(y / 2), math.sin(y / 2), math.cos(p / 2), math.sin(p / 2), math.cos(r / 2), math.sin(r / 2)
        qw = cr * cp * cy + sr * sp * sy
        qx = cr * sp * cy + sr * cp * sy
        qy = cr * cp * sy - sr * sp * cy
        qz = sr * cp * cy - cr * sp * sy
        self.c.send_message("/VMC/Ext/Bone/Pos", ["Head", 0.0, 0.0, 0.0, qx, qy, qz, qw])


class VTubeStudioClient:
    """VTube Studio plugin API: injects default tracking parameters. VTS asks the user to allow the plugin once."""

    def __init__(self, url="ws://localhost:8001", token_path=None):
        self.url = url
        self.token_path = token_path
        self.ws = None

    async def connect(self):
        import websockets
        from pathlib import Path
        self.ws = await websockets.connect(self.url)
        name, dev = "Mimetic", "Mimetic muscle face"
        tok = None
        if self.token_path and Path(self.token_path).exists():
            tok = Path(self.token_path).read_text().strip()
        if tok is None:
            r = await self._req("AuthenticationTokenRequest", {"pluginName": name, "pluginDeveloper": dev})
            tok = r["data"]["authenticationToken"]
            if self.token_path:
                Path(self.token_path).write_text(tok)
        await self._req("AuthenticationRequest", {"pluginName": name, "pluginDeveloper": dev, "authenticationToken": tok})

    async def _req(self, typ, data):
        msg = {"apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0", "requestID": typ, "messageType": typ, "data": data}
        await self.ws.send(json.dumps(msg))
        return json.loads(await self.ws.recv())

    async def send(self, params: dict):
        vals = [{"id": k, "value": float(v)} for k, v in params.items()]
        await self._req("InjectParameterDataRequest", {"faceFound": True, "mode": "set", "parameterValues": vals})
