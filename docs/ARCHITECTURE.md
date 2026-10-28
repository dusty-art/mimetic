# Mimetic: architecture

Mimetic animates a face by deciding which **muscles** fire, and lets a
physics model of skin turn those muscle forces into motion. Text (via
text-to-speech) or live audio goes in; muscle activations come out, and from
them a simulated face, ARKit blendshapes, Live2D/VTube Studio parameters, or
Blender point caches.

```
text ──► Kokoro TTS ──► audio + exact phoneme timing ─┐
                                                      ├─► speech+emotion controller ─► 52 muscle activations
mic ──► streaming audio model (80 ms lookahead) ──────┘            (47 facial/eyelid muscles + 5 jaw)
                                                                            │
                   ┌────────────────────────────────────────────────────────┤
                   ▼                       ▼                      ▼          ▼
         soft-tissue simulator /   ARKit 52 blendshapes   VTube Studio /   Blender PC2 cache,
         neural surrogate (skin)   (3D avatars, VMC)      Live2D params    Live Link CSV
```

## 1. Body: anatomy and soft-tissue physics (`mimetic/face/`)

* **Mesh.** MediaPipe's 468-vertex canonical face topology. That is the same
  topology MediaPipe tracks in ordinary video, so every tracked frame of every
  dataset is a dense correspondence to the simulated skin. The rest shape is
  the measured average neutral face of 78 real speakers, not the canonical
  mesh, whose eyes are about 40% too narrow.
* **Muscles** (`anatomy.py`). 23 bilateral muscles plus procerus, each with an
  atlas-style origin on bone (or galea / clavicle), a soft-tissue insertion,
  a FACS action unit, and a maximal excursion. Linear muscles (zygomaticus
  major/minor, levators, risorius, buccinator, depressors, mentalis,
  platysma, frontalis, corrugator…), sphincters (orbicularis oculi orbital,
  orbicularis oris peripheral with pursing/protrusion), the marginal
  orbicularis oris (lip rolling/pressing), eyelid closer and opener, and five
  jaw actuators driving a rigid mandible that rotates about the
  temporomandibular joint with condylar translation. Origins on the mandible
  move with it.
* **Simulator** (`sim.py`). Quasi-static projective dynamics. Skin stretch
  springs, smoothness of deviation from the skinned bone pose, tethering to
  bone with region-specific stiffness (retaining ligaments vs loose lips and
  cheeks), and a volume-preserving thickness constraint, so compressed tissue
  bulges (cheeks in a smile, lips in a pucker). Lip and eyelid contact and
  eyelids sliding on the eyeball. Muscle forces follow a force-length curve,
  so expressions saturate the way real faces do. Each muscle's force is
  calibrated automatically to its anatomical excursion. All weights are
  constant, so the 468×468 system matrix is inverted once and solves batch.
* **Surrogate** (`surrogate.py`). The jaw is analytic, and an MLP learns the
  soft-tissue response from 160k simulated poses. It is differentiable and
  runs in microseconds, which makes everything below trainable.

## 2. Muscle motion capture from ordinary video (`mimetic/fit/`)

A *physics-decoder autoencoder*: a temporal encoder maps tracked face
motion to muscle activations, and the frozen surrogate decodes them back to
skin. The only way to reconstruct a real face is to find muscle activations
that physically produce it. Effort (L1) and smoothness priors choose the most
economical solution, as in minimum-effort motor control. An optional
calibration layer learns a small corrective field per muscle, masked to that
muscle's anatomical territory: anatomy is the prior, real faces refine it.
The same encoder turns any webcam/video into muscle activations
("muscle mocap").

## 3. Brain: the speech + emotion controller (`mimetic/model/controller.py`)

Two pathways meet in muscle space:

* **Speech trunk.** Phonemes with articulatory features and timing, prosody
  (loudness, pitch, voicing) and a speaking-style embedding → articulation
  drive.
* **Emotion trunk.** Emotion × intensity, prosody and the speech trunk's
  state → emotional drive, gated to zero at zero intensity.
* `activations = sigmoid(tone + articulation + emotion)`. Adding drives in
  logit space gives muscle-like saturation. Because the emotion trunk sees the
  speech state, it learns to yield where speech needs the mouth: a smile
  narrows for "oo" but keeps the cheeks and eyes.

Training combines imitation of the muscle-space mocap, a skin-vertex loss
through the surrogate, and **articulation constraints**
(`articulation.py`): the lips must seal for p/b/m, the lower lip must meet
the upper teeth for f/v, the lips must round for oo/w/sh, and the jaw must
open for open vowels. They are stated from phonetics rather than learned,
hold under emotion, and are optimized by backpropagating through the
differentiable muscle face (analytic policy gradients). Plus a jerk penalty
and prosodic head motion.

## 4. Ear: live audio (`mimetic/model/audio_model.py`, `runtime/live.py`)

Causal gated dilated convolutions over 10 ms log-mel frames, with output
delayed by 8 frames (80 ms lookahead), because lips move before the sound
they shape. Trained on real recordings (with their muscle mocap) and on a
synthetic corpus of public-domain sentences read by 28 TTS voices, whose
targets come from the text controller (distillation). Noise, gain, EQ and
reverb augmentation are applied. Auxiliary heads give phoneme posteriors,
emotion from voice, and voice activity.

## 5. Life layer (`runtime/life.py`)

Behaviours that aren't driven by speech:

* **Blinks**: about 17 per minute, scaled by emotion, preferring pauses. Fast
  close, slow open.
* **Gaze**: fixations with small saccades.
* **Head drift**: band-limited, on top of the controller's prosodic head
  motion.

## 6. Outputs (`runtime/outputs.py`, `export3d.py`, `video.py`, `web/`, `blender/`)

* **ARKit 52**: learned from frames that have both a muscle fit and tracked
  blendshapes. Supports VMC (VSeeFace / Warudo / VNyan), Perfect Sync VRM,
  MetaHuman and Live Link Face CSV.
* **VTube Studio / Live2D**: parameters measured geometrically on the
  simulated face.
* **Blender**: PC2 point cache for the Mesh Cache modifier, plus jaw, teeth,
  tongue and eyes, through an add-on.
* **Web demo**: three.js, with text and emotion tags, live mic, and a muscle
  overlay.

## Data

| corpus | licence | content | use |
|---|---|---|---|
| GRID | CC BY 4.0 | 33 speakers × 1000 sentences, neutral | articulation |
| CREMA-D | ODbL / DbCL | 91 actors, 12 sentences, 6 emotions × 3 levels | emotion + speech |
| RAVDESS | CC BY-NC-SA 4.0 | 24 actors, 8 emotions × 2 levels | emotion + speech |
| Kokoro TTS corpus | Apache-2.0 model; public-domain text | 28 voices | audio-model distillation |

Every clip records its source, so a commercially clean model can be
retrained without RAVDESS.
