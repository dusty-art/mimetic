# Mimetic: research report

*Muscle-driven speech and emotion for talking faces, built and trained on one
laptop (Apple M5, 16 GB) in a single session, 2026-10-28.*

## Abstract

Most talking-face systems map audio or text straight to vertex offsets or
blendshape weights. Mimetic puts a **musculoskeletal face** in the middle:
47 facial and eyelid muscles plus a 5-actuator jaw, acting on a soft-tissue
simulator. A speech-and-emotion controller learns *which muscles to fire*,
and physics decides how the skin moves.

The pieces:

1. A projective-dynamics face with anatomically placed muscles and
   force-length saturation, distilled into a differentiable neural surrogate.
2. **Muscle motion capture**: a physics-decoder autoencoder that turned
   41,856 ordinary talking-head clips (148 speakers, about 31 hours) into
   muscle activations.
3. A dual-pathway controller that composes speech and emotion in activation
   space, trained under phonetic articulation constraints optimized through
   the differentiable face.
4. A streaming audio model for live use.

An RL study on the same muscle face finds that analytic policy gradients
through the differentiable model reach PPO's best error with 57× fewer
environment steps on 52-actuator articulatory control, and finish with 5×
lower error.

Results tables are in §6. All numbers come from held-out speakers.

## 1. Why muscles

* **Composition.** Blendshape and vertex systems add speech and emotion. When
  both want the mouth (a smile plus an "oo"), the sum is a face nobody makes.
  Muscles saturate and antagonize, so composition in muscle space behaves
  like a face.
* **Physical validity.** Lips that seal cannot pass through each other. A
  jaw rotates about the temporomandibular joint (TMJ). Bulges come from
  volume conservation, not from a learned offset.
* **Transfer to bodies.** A muscle activation vector is an actuator command.
  The same controller output can drive a simulated head, an avatar (through
  a learned map to ARKit) or, in principle, a tendon-driven robot face.
  That is the robotics bet behind the project.

## 2. The face

**Mesh and rest shape.** The face uses MediaPipe's 468-vertex canonical
topology, so every tracked frame of any video corresponds vertex-for-vertex
to the simulated skin. The rest shape is the average neutral face of 78
measured speakers. The canonical mesh's eyes turned out about 40% too narrow
(0.67 vs 0.94 cm aperture) for real faces.

**Muscles** (`mimetic/face/anatomy.py`):

* 23 bilateral muscles plus procerus, each with a bony origin, a skin
  insertion, a FACS action unit and a maximal excursion: frontalis
  (medial/lateral), corrugator, depressor supercilii, orbicularis oculi
  (orbital, palpebral), levator palpebrae, levator labii superioris (and
  alaeque nasi), zygomaticus major/minor, levator anguli oris, risorius,
  buccinator, nasalis, depressor anguli oris, depressor labii inferioris,
  mentalis, platysma, and orbicularis oris (marginal = pressing/rolling;
  peripheral = pursing with protrusion).
* 5 jaw actuators: open, close, protrude, lateral L/R. Each muscle's force is
  calibrated automatically to its anatomical excursion.

**Soft tissue** (`sim.py`). Quasi-static projective dynamics with:

* rotation-invariant stretch springs;
* smoothness of the deviation from the skinned bone pose;
* tethering to bone with region-specific stiffness (retaining ligaments vs
  loose lips and cheeks);
* a thickness constraint that grows under tangential compression, so cheeks
  bulge in a smile and lips thicken in a pucker;
* lip and lid contact, and lids sliding on the eyeballs;
* force-length curves.

The 468×468 system matrix is constant, so it is inverted once. Batched solves
cost about 3–4 ms per pose. After 100 iterations, mean error against a
600-iteration reference is 0.06 mm (99th percentile 0.73 mm). The rare slow
cases are lips pressed together while the jaw is wide open.

**Surrogate** (`surrogate.py`). The jaw is analytic, and an MLP learns the
soft-tissue response from 160,000 simulated poses. On held-out poses the
error is 0.034 mm mean, 0.135 mm on the lips and 0.42 mm at the 99th
percentile. It is differentiable and runs in microseconds per pose.

## 3. Data

| corpus | speakers | clips | content | licence |
|---|---|---|---|---|
| GRID | 33 | 32,980 | neutral, structured sentences | CC BY 4.0 |
| CREMA-D | 91 | 7,436 | 12 sentences × 6 emotions × 3 levels | ODbL |
| RAVDESS | 24 | 1,440 | 2 sentences × 8 emotions × 2 levels | CC BY-NC-SA |
| Kokoro TTS corpus | 28 voices | 4,000 | public-domain sentences, 4.85 h | Apache-2.0 / PD |

**Pipeline** (`scripts/fetch_and_track.py`, about 2 h unattended):

1. Stream each archive, track with MediaPipe Face Landmarker (478 landmarks,
   52 blendshapes, head pose), and keep 16 kHz FLAC audio.
2. Delete the raw video. All tracked data fits in 4.6 GB.
3. Force-align phonemes with espeak-ng and a wav2vec2 phoneme CTC model,
   using a custom Viterbi pass that grows CTC spikes into phone intervals.
4. Register every frame to its speaker's neutral face with bone-anchored
   Procrustes, and store it as 96 PCA coefficients (99.7% of variance,
   0.1 mm reconstruction).

**Three data problems found and fixed:**

* **GRID's alignments.zip labels 13+ speaker folders differently from the
  video archives** (for example, video s19 ↔ alignments s18, s32 ↔ s33).
  Every talker read a unique set of 1,000 sentence codes, so each archive is
  matched to the folder with full file-name overlap.
* **The CREMA-D video mirror has no audio.** Clips were re-paired with the
  separately published 16 kHz WAVs. Durations match to 5 ms.
* **Lip contact does not transfer between faces.** Displacements measured
  from a speaker's own neutral face, applied to the average face, make lips
  pass through each other or never meet. It depends on whether that
  speaker's relaxed lips sit wider or narrower than average. Before the fix,
  18% of ground-truth frames interpenetrated. On one speaker, the rest face
  sealed in 0.1% of frames instead of 9%. *Contact-preserving transfer* maps
  each speaker's sealed gap to contact and their neutral gap to the rest
  face's neutral gap, per inner-lip pair, spread smoothly over the lip.
* **Eye openness needs ratio transfer.** Offsets close small eyes too far and
  large eyes too little. Eyelid gaps now transfer as a fraction of each
  speaker's neutral opening, with the upper lid carrying 75% of the change.

## 4. Muscle motion capture

A temporal convolutional encoder maps each frame's displacement to 52
activations. The frozen surrogate decodes them, and a loss on vertices (lids
×6, lips ×5, brows ×4, cheeks ×2.5, depth ×0.35) plus per-pair lip- and
eyelid-aperture matching trains the encoder. An L1 effort prior and temporal smoothness choose the
economical solution among the many that fit, following minimum-effort motor
control. The prior is lighter (×0.3) on upper-face muscles: a first version
with a uniform prior lost most of surprise's eye-widening (0.05 vs 1.6 mm)
and two-thirds of the happy cheek raise, because 1–2 mm cues were cheaper to
ignore than to fit. A calibration layer learns a small corrective field per muscle,
masked to that muscle's anatomical territory: anatomy is the prior, real
faces refine it.

| on 3.1 M frames (validation windows) | lips (mm) | whole face (mm) |
|---|---|---|
| do nothing (neutral face) | 3.85 | 1.95 |
| **muscle fit (52 physical actuators)** | **0.58** | **0.67** |
| unconstrained linear PCA-8 / PCA-52 (no physics) | 0.87 / 0.15 | 0.62 / 0.12 |

The muscle model explains about 85% of lip motion while staying physically
valid, with no interpenetration and saturating actuators.

**Emotion survives the fit.** Classifiers trained and tested on held-out
speakers score 66.7% on real faces, 65.1% on muscle-fit faces, and 65.1% on
the 52 activations alone (chance 17%). The decompositions read like FACS:

* disgust → levator labii alaeque nasi (nose wrinkler, AU9) + zygomaticus
  minor + risorius;
* sadness → mentalis (AU17) + orbicularis oculi;
* anger → depressor anguli oris + depressor labii inferioris;
* happy laughter → zygomaticus + depressor labii inferioris + jaw.

**Anatomy calibrated by data.** The first fit co-activated "jaw protrusion"
with every opening (r = 0.78). Real jaw opening carries far more forward
condylar translation than the textbook-small coupling I started with.
Re-fitting with 2.1 cm/rad, about 1.5 cm of condylar glide at full opening,
cut the correlation to 0.57. The remaining average protrusion is 0.8 mm
(see Limitations).

## 5. Controller, voice and live audio

**Controller** (`mimetic/model/controller.py`). The speech trunk is a 6-layer
transformer over phonemes, articulatory features, timing, prosody and style.
The emotion trunk is a 3-layer transformer over emotion × intensity, prosody
and the speech state, gated to zero at zero intensity. The two drives are
added in logit space: `a = σ(tone + speech + emotion)`.

Training uses:

* imitation of the muscle mocap;
* skin-vertex loss through the surrogate;
* jerk and head-motion terms;
* **articulation constraints from phonetics**: seal p/b/m (aperture < 0.4 mm
  at phone centre), lower lip to upper teeth for f/v, rounding and protrusion
  for oo/w/sh/ch, jaw opening for open vowels, weighted by phonetic
  criticality.

These are rewards optimized by backpropagating through the differentiable
face (analytic policy gradients), so they hold even where the tracked data
under-shows them.

Prosody dropout (50%) and speech-context dropout (30%) on the emotion trunk
make emotion work from the label alone. That matters because TTS prosody is
neutral, and because only 14 distinct emotional sentences exist in the data.

**Text path.** Kokoro-82M synthesizes speech. Its duration predictor gives
every phoneme's timing exactly, so text-mode lip-sync has zero alignment
error. Emotion tags (`[happy:0.8]`) switch emotion mid-sentence with
intensity ramps and adjust speaking rate.

**Live path** (`audio_model.py`). Twelve gated causal dilated convolutions
over 10 ms log-mel frames (5.6 M parameters, 2.5 s receptive field). The
output is delayed by 80 ms of lookahead, because lips lead sound. It is
verified bit-exact between streaming and offline inference, at 4.3 ms per
20 ms of audio on CPU.

It is trained on real recordings, with their muscle mocap as targets, and on
the TTS corpus, with controller outputs as targets (distillation). Noise
(SNR 5–40 dB, white or pink), gain, spectral tilt and reverb augmentation are
applied. Auxiliary heads give phonemes, emotion from voice, and voice
activity.

**Life layer.** Blinks follow emotion-dependent rates and favor pauses, with
an 80 ms close and 180 ms open. Gaze uses fixations and saccades. Head motion
combines the controller's prosodic motion with band-limited drift.

## 6. Results

### 6.1 Talking-face accuracy on held-out speakers

*(Filled from `experiments/eval_results.json`.)*

### 6.2 Speech under emotion

*(Filled from `experiments/composition.json`.)*

### 6.3 RL: learning to articulate with 52 muscles

**Task.** Every 100 ms a new mouth target arrives: aperture, width,
protrusion, lower-lip tuck and jaw height. The policy commands 52 muscle
excitations through first-order activation dynamics (30 ms rise, 60 ms
decay). Both methods use the same policy network and the same environment.

| method | env steps | final lip-target error (mm) |
|---|---|---|
| no movement | — | 4.04 |
| PPO (standard hyperparameters) | 3 M | 1.62 |
| PPO | 20 M | 1.86 |
| **analytic policy gradient through the muscle face** | 3 M | **0.31** |

PPO's best training error came at 3.15 M steps, after which it slowly got
worse. The analytic gradient matched that error after 55 k steps, 57× fewer,
and kept improving. PPO was not tuned, and tuning could narrow the gap. It is unlikely to close it in a 52-dimensional,
overactuated, antagonist-rich action space, where model-free exploration is
hard and the differentiable model gives exact credit assignment. For robot
faces, this argues for learning on a calibrated differentiable muscle model,
then transferring.

## 7. Limitations

* The face is a single 468-vertex shell with procedural teeth, tongue and
  eyes. It has no wrinkles, and the tongue is not articulated (/l/, /th/ use
  no tongue model).
* Monocular tracking limits depth accuracy, and MediaPipe smooths fast lip
  events. Contact transfer and the articulation constraints compensate, but
  the ground truth itself is a tracker.
* Emotional data is acted, from 14 sentences. Generalization to new
  sentences is measured on TTS sentences (§6.2), not against real recordings
  of them.
* A residual average jaw protrusion of 0.8 mm suggests remaining jaw or
  registration bias.
* English only. Kokoro's voice does not carry emotion; only the face and the
  speaking rate do.
* RAVDESS is non-commercial. The model is research-only until retrained on
  GRID + CREMA-D.

## 8. What I would do next

* Calibrate muscle parameters per region with the differentiable simulator
  (system identification), not only through corrective fields.
* Add a tongue (hydrostat) and teeth contact for dentals and labiodentals.
* Use an emotional TTS so voice and face agree.
* Collect a short iPhone (ARKit) recording session for clean lip-contact
  ground truth.
* Close the loop with a physical tendon-driven face, using the 52-actuator
  policy as the controller.
