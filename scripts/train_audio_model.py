"""Train the streaming audio -> muscle model on real recordings (muscle mocap targets) and the TTS corpus
(teacher targets from the text controller), with acoustic augmentation.

  python scripts/train_audio_model.py --steps 20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mimetic.data import prosody, track  # noqa: E402
from mimetic.face import anatomy as A  # noqa: E402
from mimetic.model.articulation import ArticulationLoss  # noqa: E402
from mimetic.model.audio_model import HOP, N_FFT, SR, AudioToMuscle  # noqa: E402
from mimetic.model.controller import EMOTIONS, SpeechEmotionController  # noqa: E402
from mimetic.paths import CHECKPOINTS, DATA, EXPERIMENTS, PROCESSED  # noqa: E402
from mimetic.runtime.engine import emotion_track, phone_track  # noqa: E402
from mimetic.runtime.face_model import load_face_model  # noqa: E402
from train_controller import TEST_SPEAKERS, intensity_of  # noqa: E402

CROP_S = 2.5
FPS = 30.0
CACHE = DATA / "audio_cache"


def build_real_cache():
    """Concatenate 16 kHz audio of every controller clip into one int16 memmap."""
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA / "ctrl" / "clips.json").read_text())
    idx_p = CACHE / "real_index.json"
    if idx_p.exists():
        return json.loads(idx_p.read_text())
    chunks, index, off = [], [], 0
    for c in meta["clips"]:
        p = PROCESSED / c["dataset"] / c["speaker"] / f"{c['clip']}.npz"
        try:
            a = track.load_clip(p)["audio"]
        except Exception:
            continue
        if len(a) < SR:
            continue
        q = np.clip(a * 32767, -32768, 32767).astype(np.int16)
        chunks.append(q)
        index.append({"a_off": off, "a_n": len(q), "f_off": c["offset"], "f_n": c["n"], "speaker": c["speaker"],
                      "emotion": c["emotion"], "inten": intensity_of(c), "dataset": c["dataset"]})
        off += len(q)
    mm = np.lib.format.open_memmap(CACHE / "real_audio.npy", mode="w+", dtype=np.int16, shape=(off,))
    o = 0
    for q in chunks:
        mm[o:o + len(q)] = q
        o += len(q)
    mm.flush()
    idx_p.write_text(json.dumps(index))
    return index


@torch.no_grad()
def build_tts_teacher(dev):
    """Teacher activations for every TTS utterance from the trained text controller."""
    out_p = DATA / "tts" / "teacher.npz"
    lines = [json.loads(l) for l in (DATA / "tts" / "index.jsonl").open()] if (DATA / "tts" / "index.jsonl").exists() else []
    if out_p.exists():
        z = np.load(out_p, allow_pickle=False)
        if len(z["offsets"]) - 1 >= len(lines):
            return z
    ck = torch.load(CHECKPOINTS / "controller.pt", map_location="cpu", weights_only=False)
    ctrl = SpeechEmotionController(n_speakers=ck["n_speakers"]).to(dev).eval()
    ctrl.load_state_dict(ck["model"])
    acts_all, phones_all, offs, audio_all, aoffs, emos, intens = [], [], [0], [], [0], [], []
    for u in lines:
        p = DATA / "tts" / f"{u['id']}.flac"
        if not p.exists():
            continue
        a, _ = sf.read(p, dtype="float32")
        n = int(np.ceil(len(a) / SR * FPS))
        segs = [tuple(s) for s in u["segments"]]
        ids, pos, dur = phone_track(segs, n)
        pr = prosody.features(a, fps=FPS, n=n)
        e, i = emotion_track([(0.0, len(a) / SR, u["emotion"], u["intensity"])], n)
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)[None]
        acts, _ = ctrl(t(ids, torch.long), t(pos), t(dur), t(pr), t(e, torch.long), t(i),
                       torch.tensor([ck["n_speakers"]], device=dev))
        acts_all.append(acts[0].cpu().numpy().astype(np.float16))
        phones_all.append(ids.astype(np.int8))
        offs.append(offs[-1] + n)
        audio_all.append(np.clip(a * 32767, -32768, 32767).astype(np.int16))
        aoffs.append(aoffs[-1] + len(a))
        emos.append(EMOTIONS.index(u["emotion"]))
        intens.append(u["intensity"])
    np.savez(out_p, acts=np.concatenate(acts_all), phone=np.concatenate(phones_all), offsets=np.array(offs),
             audio=np.concatenate(audio_all), aoffsets=np.array(aoffs), emo=np.array(emos), inten=np.array(intens, np.float32))
    return np.load(out_p)


def interp_frames(x, n_out, fps_in=FPS, fps_out=100.0, offset_s=N_FFT / 2 / SR):
    t_out = np.arange(n_out) / fps_out + offset_s
    t_in = np.arange(len(x)) / fps_in
    return np.stack([np.interp(t_out, t_in, x[:, k]) for k in range(x.shape[1])], -1)


def nearest_frames(x, n_out, fps_in=FPS, fps_out=100.0, offset_s=N_FFT / 2 / SR):
    i = np.clip(((np.arange(n_out) / fps_out + offset_s) * fps_in).astype(int), 0, len(x) - 1)
    return x[i]


class Batcher:
    def __init__(self, split, rng):
        self.rng = rng
        z = np.load(DATA / "ctrl" / "dataset.npz")
        self.acts, self.phone, self.blink = z["acts"], z["phone"], z["blink"]
        index = build_real_cache()
        self.real = [c for c in index if (c["speaker"] in TEST_SPEAKERS) == (split == "test")]
        self.audio = np.load(CACHE / "real_audio.npy", mmap_mode="r")
        self.tts = None

    def attach_tts(self, z):
        self.tts = {k: z[k] for k in z.files}
        self.n_tts = len(self.tts["offsets"]) - 1

    def crop(self, wav, acts, phone, blink):
        S = int(CROP_S * SR)
        n_mel = (S - N_FFT) // HOP + 1
        # Audio can run a little past the last video frame: crop only where both exist.
        usable = min(len(wav), int(len(acts) / FPS * SR))
        if len(wav) < S:
            wav = np.pad(wav, (0, S - len(wav)))
        st = int(self.rng.integers(0, max(0, usable - S) + 1))
        w = wav[st:st + S]
        sh = st / SR
        def sub(x, nearest=False):
            t0 = min(int(sh * FPS), len(x) - 1)
            xx = x[t0:]
            return (nearest_frames if nearest else interp_frames)(xx.reshape(len(xx), -1), n_mel, offset_s=N_FFT / 2 / SR + (sh - t0 / FPS))
        a = sub(acts)
        ph = sub(phone[:, None], nearest=True)[:, 0]
        bl = sub(blink[:, None].astype(np.float32), nearest=True)[:, 0] > 0.5
        valid = np.arange(n_mel) * HOP + N_FFT // 2 < min(len(wav), usable) - st
        return w.astype(np.float32) / 32767, a, ph, bl, valid

    def sample(self, B, p_tts=0.4):
        W, Acts, Ph, Bl, V, E, I, syn = [], [], [], [], [], [], [], []
        for _ in range(B):
            use_tts = self.tts is not None and self.rng.random() < p_tts
            if use_tts:
                k = int(self.rng.integers(0, self.n_tts))
                o0, o1 = self.tts["offsets"][k], self.tts["offsets"][k + 1]
                a0, a1 = self.tts["aoffsets"][k], self.tts["aoffsets"][k + 1]
                wav = self.tts["audio"][a0:a1]
                acts = self.tts["acts"][o0:o1].astype(np.float32)
                phone = self.tts["phone"][o0:o1]
                blink = np.zeros(len(acts), bool)
                e, i = int(self.tts["emo"][k]), float(self.tts["inten"][k])
            else:
                c = self.real[int(self.rng.integers(0, len(self.real)))]
                wav = np.asarray(self.audio[c["a_off"]:c["a_off"] + c["a_n"]])
                acts = self.acts[c["f_off"]:c["f_off"] + c["f_n"]].astype(np.float32)
                phone = self.phone[c["f_off"]:c["f_off"] + c["f_n"]]
                blink = self.blink[c["f_off"]:c["f_off"] + c["f_n"]]
                e, i = EMOTIONS.index(c["emotion"]), c["inten"]
            w, a, ph, bl, v = self.crop(wav, acts, phone, blink)
            W.append(w); Acts.append(a); Ph.append(ph); Bl.append(bl); V.append(v); E.append(e); I.append(i)
            syn.append(use_tts)
        return {"wav": np.stack(W), "acts": np.stack(Acts).astype(np.float32), "phone": np.stack(Ph).astype(np.int64),
                "blink": np.stack(Bl), "valid": np.stack(V).astype(np.float32), "emo": np.array(E),
                "inten": np.array(I, np.float32), "syn": np.array(syn)}


def augment(wav, rng_t):
    B, S = wav.shape
    dev = wav.device
    gain = 10 ** (torch.empty(B, 1, device=dev).uniform_(-18, 6) / 20)
    x = wav * gain
    # EQ: random tilt in the spectrum
    if True:
        Xf = torch.fft.rfft(x)
        f = torch.linspace(0, 1, Xf.shape[1], device=dev)[None]
        tilt = torch.empty(B, 1, device=dev).uniform_(-6, 6)
        Xf = Xf * 10 ** ((tilt * (f - 0.3)) / 20)
        x = torch.fft.irfft(Xf, n=S)
    # Reverb
    rev = torch.rand(B, device=dev) < 0.3
    if rev.any():
        L = 4000
        t = torch.arange(L, device=dev) / SR
        rt = torch.empty(B, 1, device=dev).uniform_(0.15, 0.6)
        ir = torch.randn(B, L, device=dev) * torch.exp(-6.9 * t / rt)
        ir[:, 0] = 1.0
        ir = ir / ir.norm(dim=1, keepdim=True)
        y = torch.fft.irfft(torch.fft.rfft(x, n=S + L) * torch.fft.rfft(ir, n=S + L), n=S + L)[:, :S]
        x = torch.where(rev[:, None], y, x)
    # Noise
    noisy = torch.rand(B, device=dev) < 0.7
    snr = torch.empty(B, 1, device=dev).uniform_(5, 40)
    n = torch.randn(B, S, device=dev)
    pink = torch.rand(B, device=dev) < 0.5
    if pink.any():
        Nf = torch.fft.rfft(n)
        f = torch.arange(Nf.shape[1], device=dev).clamp_min(1)
        n = torch.where(pink[:, None], torch.fft.irfft(Nf / f.sqrt(), n=S), n)
    ps = x.pow(2).mean(1, keepdim=True).clamp_min(1e-8)
    pn = n.pow(2).mean(1, keepdim=True).clamp_min(1e-8)
    n = n * torch.sqrt(ps / pn / 10 ** (snr / 10))
    x = torch.where(noisy[:, None], x + n, x)
    return x.clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lookahead", type=int, default=8)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = Batcher("train", rng)
    tts = build_tts_teacher(dev) if (DATA / "tts" / "index.jsonl").exists() else None
    if tts is not None:
        data.attach_tts(tts)
        print(f"TTS corpus: {len(tts['offsets']) - 1} utterances", flush=True)
    print(f"real clips: {len(data.real)}", flush=True)
    face = load_face_model(dev)
    model = AudioToMuscle(lookahead=args.lookahead).to(dev)
    # Global log-mel statistics
    with torch.no_grad():
        b = data.sample(64, p_tts=0.3)
        mel = model.mel(torch.tensor(b["wav"], device=dev))
        model.mel_mean.copy_(mel.mean((0, 1)))
        model.mel_std.copy_(mel.std((0, 1)).clamp_min(1e-3))
    art = ArticulationLoss(face.rest, A.MUSCLE_NAMES.index("jaw_open"))
    pal = [A.MUSCLE_NAMES.index(f"orbicularis_oculi_palpebral_{s}") for s in "RL"]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)
    L = args.lookahead
    t0 = time.time()
    log = []
    for step in range(1, args.steps + 1):
        b = data.sample(args.bs)
        tb = {k: torch.as_tensor(v, device=dev) for k, v in b.items()}
        wav = augment(tb["wav"], None)
        auto = (torch.rand(args.bs, device=dev) < 0.5) & ~tb["syn"]
        emo_in = torch.where(auto, torch.full_like(tb["emo"], len(EMOTIONS)), tb["emo"])
        out = model(wav, emo_in, tb["inten"])
        T = out["acts"].shape[1]
        acts = out["acts"][:, L:]                             # output delayed by L frames
        tgt = tb["acts"][:, :T - L]
        ph = tb["phone"][:, :T - L]
        m = tb["valid"][:, :T - L]
        w = torch.ones_like(tgt)
        w[..., pal] = torch.where(tb["blink"][:, :T - L, None], 0.0, 1.0)
        l_act = ((F.smooth_l1_loss(acts, tgt, beta=0.05, reduction="none") * w).mean(-1) * m).sum() / m.sum()
        sub = acts[:, ::4]
        x = face(sub.reshape(-1, A.N_ACT)).view(*sub.shape[:2], -1, 3)
        with torch.no_grad():
            xt = face(tgt[:, ::4].reshape(-1, A.N_ACT)).view(*sub.shape[:2], -1, 3)
        ms = m[:, ::4]
        l_vert = ((((x - xt) ** 2).sum(-1).mean(-1)) * ms).sum() / ms.sum()
        ppos = torch.full_like(ms, 0.5)
        l_art, st = art(x, sub, ph[:, ::4], ppos, ms)
        l_ph = (F.cross_entropy(out["phone"][:, L:].transpose(1, 2), ph, reduction="none") * m).sum() / m.sum()
        speech = (ph > 0).float() * m
        real = (~tb["syn"]).float()[:, None]
        l_emo = (F.cross_entropy(out["emotion"][:, L:].transpose(1, 2), tb["emo"][:, None].expand_as(ph),
                                 reduction="none") * speech * real).sum() / (speech * real).sum().clamp_min(1)
        l_vad = (F.binary_cross_entropy_with_logits(out["vad"][:, L:], (ph > 0).float(), reduction="none") * m).sum() / m.sum()
        d1 = acts[:, 2:] - 2 * acts[:, 1:-1] + acts[:, :-2]
        l_sm = ((d1 ** 2).sum(-1) * m[:, 2:]).sum() / m.sum()
        loss = l_act + 2.0 * l_vert + 0.5 * l_art + 0.2 * l_ph + 0.1 * l_emo + 0.1 * l_vad + 2.0 * l_sm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 250 == 0:
            acc = ((out["phone"][:, L:].argmax(-1) == ph).float() * speech).sum() / speech.sum().clamp_min(1)
            rec = {"step": step, "loss": loss.item(), "act": l_act.item(), "vert": l_vert.item(), "art": l_art.item(),
                   "phone_acc": acc.item(), "emo": l_emo.item(), "t": time.time() - t0}
            log.append(rec)
            print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
        if step % 5000 == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "config": {"lookahead": L}}, CHECKPOINTS / "audio_model.pt")
    (EXPERIMENTS / "audio_model_trainlog.json").write_text(json.dumps(log))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
