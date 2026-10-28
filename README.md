# Mimetic

**Muscle-driven talking faces.** Type a sentence, or talk into a microphone, and
Mimetic decides which of 47 facial muscles and 5 jaw actuators fire, frame by
frame. A soft-tissue physics model then turns those muscle forces into skin
motion. Speech and emotion meet in muscle space: a smile yields to an "oo"
the way a real face does, and lips really seal on "p", "b" and "m".

The output drives a simple simulated head, 3D avatars (ARKit 52 blendshapes,
VMC protocol), Live2D / VTube Studio, and Blender.

> Status: research prototype, trained on a laptop (Apple M5, 16 GB). Results
> are in [docs/REPORT.md](docs/REPORT.md); the design is in
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

```bash
git clone https://github.com/dusty-art/mimetic && cd mimetic
```

```bash
uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e ".[dev]"
```

Download the trained models (108 MB, from the
[v0.1.0 release](https://github.com/dusty-art/mimetic/releases/tag/v0.1.0)):

```bash
curl -L https://github.com/dusty-art/mimetic/releases/download/v0.1.0/mimetic-models-v0.1.0.tar.gz | tar xz
```

You also need `uv`, `espeak-ng` and `ffmpeg` (`brew install uv espeak-ng ffmpeg` on
macOS). The first run downloads the Kokoro voice from Hugging Face. The model
weights are CC BY-NC-SA 4.0, for non-commercial use (see
`checkpoints/LICENSE.md`). The datasets are not redistributed; to retrain, follow
[Reproducing](#reproducing).

Speak a line and render a video (emotion tags apply from that point on):

```bash
mimetic say "[happy:0.8] Hi! Every movement is a muscle. [sad:0.6] Even this one." -o hello.mp4 --muscles
```

Web demo (text, emotion buttons, live mic, muscle overlay), at http://127.0.0.1:8765:

```bash
mimetic serve
```

Export for other tools (JSON, Live Link CSV, WAV, OBJ + PC2 point cache):

```bash
mimetic export --text "[surprised] Oh! You're here." -o out/
```

```bash
mimetic export --audio speech.wav --transcript "what was said" -o out/
```

Muscle motion capture from any face video (retargeted through muscle space):

```bash
mimetic mocap my_face.mp4 -o out/ --render
```

Exaggerate emotion for cartoony avatars without distorting lip-sync:

```bash
mimetic say "[surprised] No way!" --expressiveness 1.8 -o wow.mp4
```

Live microphone to a VTuber app:

```bash
mimetic live --vmc
```

```bash
mimetic live --vts
```

## VTubers and 3D avatars

Step-by-step setup for each app: [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md).

* **VSeeFace, Warudo, VNyan** (VMC protocol): enable the VMC receiver on port
  39539 and run `mimetic live --vmc`. Mimetic sends the 52 ARKit blendshapes,
  so "Perfect Sync" VRM avatars get the full face. It also sends head rotation.
* **VTube Studio / Live2D**: run `mimetic live --vts` and approve the plugin
  in VTube Studio once. It injects MouthOpen, MouthSmile, EyeOpen L/R,
  BrowLeftY/RightY, FaceAngle X/Y/Z, eye gaze, MouthX and FaceAngry,
  measured on the simulated face.
* **Unreal / MetaHuman / anything that reads Live Link Face CSV**: use
  `livelink.csv` from `mimetic export`.

Live mode uses a causal audio model with 80 ms lookahead. It needs 3 ms of
CPU per 20 ms of audio, and the face trails your voice by about 135 ms.
Emotion comes from the voice (right about half the time on unseen speakers).
To set it yourself, use `--emotion happy --intensity 0.8`.

## Results (15 held-out speakers)

| | lip error (mm) | p/b/m seals | seals under anger | emotion recognized | lips interpenetrating |
|---|---|---|---|---|---|
| viseme table | 4.48 | 0% | 0% | 25% | 0% |
| same network, no muscle layer | 3.97 | 100% | 81% | 90% | 0.58% |
| **Mimetic, text** | 4.00 | **100%** | **100%** | 75% | **0.004%** |
| Mimetic, live mic | 4.30 | 97% | — | 35% | 0.004% |

The tracked real faces show seals on 31% of p/b/m sounds; the tracker misses
most lip contacts. For an RL controller on the same 52-muscle face, analytic
gradients reach PPO's best error with 57× fewer steps. Details, ablations and
limitations: [docs/REPORT.md](docs/REPORT.md).

## Blender

1. Zip `blender/mimetic_blender`, then install it via Edit › Preferences ›
   Add-ons › Install.
2. In the 3D View sidebar's **Mimetic** tab, set *Mimetic project* to this
   folder, enter text (or an audio file), and click **Generate**.
3. Pick a target:
   * **Mimetic head** imports the muscle-simulated head: skin through a Mesh
     Cache (PC2) modifier, plus jaw, teeth, tongue, and eyes with gaze. The
     voice is added to the sequencer.
   * **Selected mesh (ARKit shape keys)** keyframes the 52 ARKit shape keys on
     your own character.

## How it works (one paragraph)

The face is MediaPipe's 468-vertex topology with an anatomical muscle layer:
origins on bone, insertions in skin, FACS actions. A projective-dynamics
soft-tissue solver handles skin elasticity, volume conservation, lip/lid
contact, force-length saturation, and a TMJ jaw. A neural surrogate makes it
differentiable. A physics-decoder autoencoder turned roughly 30 hours of
ordinary talking-head video (GRID, CREMA-D, RAVDESS) into **muscle motion
capture**. A dual-pathway controller (speech trunk + emotion trunk, combined
in activation space) learns from it, under articulation constraints from
phonetics that are optimised through the differentiable face. Text goes
through Kokoro TTS for exact phoneme timing. Live audio goes through a
streaming causal convolutional network. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Reproducing

```bash
python scripts/fetch_and_track.py
```

```bash
python scripts/align_all.py
```

```bash
python scripts/build_motion.py
```

```bash
python scripts/train_surrogate.py
```

```bash
python scripts/fit_muscles.py --calibrate
```

```bash
python scripts/build_ctrl_dataset.py
```

```bash
python scripts/train_controller.py
```

```bash
python scripts/make_tts_corpus.py && python scripts/train_audio_model.py && python scripts/fit_live_post.py
```

```bash
python scripts/train_arkit_map.py && python scripts/evaluate.py
```

```bash
python scripts/composition_test.py && python scripts/emotion_cues.py && python scripts/figures.py
```

## Data and licences

| corpus | licence | used for |
|---|---|---|
| GRID (Zenodo 3625687) | CC BY 4.0 | neutral articulation, 33 speakers |
| CREMA-D | ODbL / DbCL | emotional speech, 91 actors |
| RAVDESS (Zenodo 1188976) | CC BY-NC-SA 4.0 | emotional speech, 24 actors |
| Kokoro-82M | Apache 2.0 | text-to-speech voice and timing |
| wav2vec2-lv-60-espeak-cv-ft | Apache 2.0 | phoneme forced alignment |
| MediaPipe Face Landmarker | Apache 2.0 | face tracking, canonical mesh |
| Project Gutenberg texts | public domain | synthetic speech corpus |

Because RAVDESS is non-commercial and share-alike, the released model weights are
CC BY-NC-SA 4.0: research and other non-commercial use.
Every clip records its source, so a commercially clean model can be retrained
on GRID + CREMA-D only.

## Licence

* **Code:** MIT, see [LICENSE](LICENSE). This covers everything in the
  repository except the files listed below.
* **CC BY-NC-SA 4.0** applies to the model weights in the
  [release](https://github.com/dusty-art/mimetic/releases/tag/v0.1.0) and to
  files derived from the datasets or those models: `assets/rest_neutral.npy`
  (the measured mean neutral face) and the figures, demos and report under
  `experiments/`.
* **Apache 2.0** (Google MediaPipe) applies to `assets/face_landmarker.task` and
  `assets/canonical_face_model.obj`.
