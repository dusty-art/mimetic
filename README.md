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
cd ~/Desktop/test && source .venv/bin/activate
```

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

Live microphone to a VTuber app:

```bash
mimetic live --vmc
```

```bash
mimetic live --vts
```

## VTubers and 3D avatars

* **VSeeFace, Warudo, VNyan** (VMC protocol): enable the VMC receiver on port
  39539 and run `mimetic live --vmc`. Mimetic sends the 52 ARKit blendshapes,
  so "Perfect Sync" VRM avatars get the full face. It also sends head rotation.
* **VTube Studio / Live2D**: run `mimetic live --vts` and approve the plugin
  in VTube Studio once. It injects MouthOpen, MouthSmile, EyeOpen L/R,
  BrowLeftY/RightY, FaceAngle X/Y/Z, eye gaze, MouthX and FaceAngry,
  measured on the simulated face.
* **Unreal / MetaHuman / anything that reads Live Link Face CSV**: use
  `livelink.csv` from `mimetic export`.

Live mode uses a causal audio model with 80 ms lookahead. It runs about 5×
faster than real time on the CPU. Emotion comes from the voice, or you can
fix it with `--emotion happy --intensity 0.8`.

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
python scripts/make_tts_corpus.py && python scripts/train_audio_model.py
```

```bash
python scripts/train_arkit_map.py && python scripts/evaluate.py
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

Because RAVDESS is non-commercial, the released model is for research use.
Every clip records its source, so a commercially clean model can be retrained
on GRID + CREMA-D only.
