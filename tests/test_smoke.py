"""Smoke tests for the core pieces. Run: python -m pytest tests -q"""
import numpy as np
import pytest
import torch

from mimetic.data import phonemes as P
from mimetic.face import anatomy as A
from mimetic.face import canonical as C
from mimetic.paths import CHECKPOINTS


def test_canonical_mesh_has_mouth_and_eye_holes():
    f = C.load_canonical()
    assert f.verts.shape == (468, 3)
    assert len(f.skin_faces) == len(f.faces) - 18 - 14 - 14
    mi = f.mirror_index()
    assert np.allclose(f.verts[mi] * [-1, 1, 1], f.verts)


def test_anatomy_is_bilateral_and_complete():
    names = A.MUSCLE_NAMES
    assert len(names) == A.N_ACT == 52
    for n in names:
        if n.endswith("_R"):
            assert n[:-2] + "_L" in names


def test_sim_rest_and_single_muscle_response():
    from mimetic.face.sim import FaceSim
    sim = FaceSim(calibrate=False, iters=40)
    x0 = sim.solve(sim.neutral_act())
    assert torch.isfinite(x0).all()
    a = sim.act_from_dict({"zygomaticus_major_R": 1.0})
    x = sim.solve(a)
    corner = C.NAMED["mouth_corner_R"]
    assert x[0, corner, 1] > x0[0, corner, 1]          # smile lifts the corner
    a = sim.act_from_dict({"jaw_open": 0.5})
    x = sim.solve(a)
    assert (x[0, 13, 1] - x[0, 14, 1]) > (x0[0, 13, 1] - x0[0, 14, 1]) + 0.5   # mouth opens


@pytest.mark.skipif(not (CHECKPOINTS / "surrogate.pt").exists(), reason="needs trained surrogate")
def test_surrogate_matches_simulator():
    from mimetic.face.sim import FaceSim
    from mimetic.face.surrogate import sample_activations
    from mimetic.runtime.face_model import load_face_model
    sim = FaceSim(calibrate=False)
    sim.F = torch.load(CHECKPOINTS / "sim_calibration.pt")["F"]
    sur = load_face_model("cpu", calibrated=False)
    a = sample_activations(64, torch.Generator().manual_seed(7))
    err = (sim.solve(a, iters=120) - sur(a)).norm(dim=-1)
    assert err.mean() < 0.02                              # < 0.2 mm mean


def test_phoneme_front_ends_cover_inventory():
    assert P.espeak_to_unified("tʃ") == ["CH"]
    assert P.espeak_to_unified("ɑːɹ") == ["AA", "R"]
    assert P.misaki_to_unified("ʤ") == "JH"
    assert P.misaki_to_unified("ˈ") is None
    assert P.ARTIC.shape == (P.N_PHONES, 7)
    assert P.ARTIC[P.PID["M"], 0] == 1.0                  # bilabials must close


def test_tts_tags_and_tracks():
    from mimetic.runtime.engine import emotion_track, phone_track
    from mimetic.runtime.tts import parse_tags
    chunks = parse_tags("[happy:0.6] Hi there. [sad] Bye.")
    assert chunks == [("Hi there.", "happy", 0.6), ("Bye.", "sad", 0.8)]
    ids, pos, dur = phone_track([(0.0, 0.1, "M"), (0.1, 0.3, "AA")], 12)
    assert ids[1] == P.PID["M"] and ids[5] == P.PID["AA"]
    e, i = emotion_track([(0.0, 0.2, "happy", 0.8), (0.2, 0.4, "sad", 0.8)], 12)
    assert e[2] != e[10] and i.min() >= 0


def test_controller_and_audio_model_shapes():
    from mimetic.model.audio_model import AudioToMuscle
    from mimetic.model.controller import SpeechEmotionController
    c = SpeechEmotionController(n_speakers=3, d=64, layers=2, d_emo=32, emo_layers=1)
    B, T = 2, 40
    acts, head = c(torch.zeros(B, T, dtype=torch.long), torch.rand(B, T), torch.rand(B, T), torch.rand(B, T, 4),
                   torch.tensor([1, 2]), torch.tensor([0.5, 1.0]), torch.tensor([0, 3]))
    assert acts.shape == (B, T, A.N_ACT) and head.shape == (B, T, 3)
    assert (acts >= 0).all() and (acts <= 1).all()
    m = AudioToMuscle(ch=32, dilations=(1, 2, 4)).eval()
    wav = torch.randn(1, 8000) * 0.1
    e, i = torch.tensor([9]), torch.tensor([1.0])
    with torch.no_grad():
        full = m(wav, e, i)["acts"]
        mel = m.mel(wav)
        st = m.init_state()
        parts = []
        for k in range(0, mel.shape[1], 3):
            o, st = m.stream(mel[:, k:k + 3], e, i, st)
            parts.append(o["acts"])
    assert torch.allclose(torch.cat(parts, 1), full, atol=1e-5)    # streaming == offline (causal)


def test_exports(tmp_path):
    from mimetic.runtime import export3d
    frames = np.random.rand(3, 10, 3).astype(np.float32)
    export3d.write_pc2(tmp_path / "x.pc2", frames, 30)
    raw = (tmp_path / "x.pc2").read_bytes()
    assert raw[:12] == b"POINTCACHE2\0" and len(raw) == 32 + frames.nbytes
    M = export3d.jaw_tracks(np.zeros((4, A.N_ACT), np.float32))
    assert np.allclose(M[0], np.eye(4), atol=1e-6)


def test_lip_and_lid_transfer_preserve_contact_and_ratio():
    from mimetic.data import motion as MO
    rest = np.load("assets/rest_neutral.npy")
    N = rest.copy()
    N[MO.LIP_U, 1] += 0.2                       # this speaker's relaxed lips sit 2 mm wider apart
    closed = MO.pair_apertures(rest[None])[0] * 0 + 0.0
    neutral = MO.pair_apertures(N[None])[0]
    seal = np.zeros((1, 468, 3))
    seal[0, MO.LIP_U, 1] = -neutral              # the speaker seals their lips
    out = MO.apply_lip_contact(seal, N, closed, neutral, MO.pair_apertures(rest[None])[0])
    ap = MO.pair_apertures(rest[None] + out)[0]
    assert np.abs(ap).max() < 0.02               # sealed on the rest face too, not interpenetrating
    half = np.zeros((1, 468, 3))
    g = MO.lid_apertures(N[None])[0]
    half[0, MO.LID_U, 1] = -0.5 * g              # eyes half closed
    out = MO.apply_lid_transfer(half, N, rest)
    ratio = MO.lid_apertures(rest[None] + out)[0] / MO.lid_apertures(rest[None])[0]
    assert np.allclose(ratio, 0.5, atol=0.05)
