# Using Mimetic with avatars and tools

Mimetic produces 52 muscle activations per frame, plus head rotation and gaze.
Everything below is derived from those.

## Live VTubing

### VSeeFace / Warudo / VNyan (3D, VMC protocol)

1. In VSeeFace: Settings → General → enable **OSC/VMC receiver**, port
   `39539`. Warudo and VNyan have the same option under their VMC receiver.
2. For full expressions, use a model with the 52 ARKit ("Perfect Sync")
   blendshapes. Models with only A/I/U/E/O still get the mouth shapes.
3. Run this, then speak:

```bash
mimetic live --vmc 127.0.0.1:39539
```

Emotion is inferred from your voice. To pin it to one emotion:

```bash
mimetic live --vmc --emotion happy --intensity 0.7
```

### VTube Studio (Live2D)

1. Run the command below. VTube Studio shows a plugin permission request
   once; click **Allow**. The token is saved to `~/.mimetic_vts_token`.
2. Mimetic sets the default tracking inputs: `MouthOpen`, `MouthSmile`,
   `EyeOpenLeft/Right`, `BrowLeftY/RightY`, `Brows`, `FaceAngleX/Y/Z`,
   `EyeLeftX/Y`, `EyeRightX/Y`, `MouthX` and `FaceAngry`. Any Live2D model
   rigged to those parameters responds.

```bash
mimetic live --vts
```

Latency: about 80 ms of model lookahead, plus your audio buffer. If your
stream's audio and video drift apart, delay the microphone in OBS by about
100 ms.

## Pre-recorded lines

* **From text:** `mimetic export --text "[happy:0.8] Hello!" -o out/`. The
  voice is synthesized with Kokoro, and the face uses its exact timing.
* **From your own recording:**
  `mimetic export --audio line.wav --transcript "what I said" -o out/`. With
  a transcript, phonemes are force-aligned. Without one, the live audio
  model is used offline.
* **From a face video (muscle mocap):** `mimetic mocap clip.mp4 -o out/ --render`.
  Your performance is retargeted through muscle space, so it stays
  physically valid on the target face.

Each export writes:

| file | use |
|---|---|
| `animation.json` | muscle activations, ARKit weights, head, gaze, phonemes and emotion spans per frame |
| `livelink.csv` | Live Link Face format: Unreal / MetaHuman, Blender Live Link importers, iClone |
| `head.obj` + `head.pc2` | the simulated head and its per-frame skin (Mesh Cache modifier in Blender, Maya, 3ds Max) |
| `tongue.obj` + `tongue.pc2` | the articulated tongue, jaw-relative |
| `jaw.npy`, `gaze.npy`, `head_rotation.npy` | rigid tracks |
| `audio.wav` | the voice |

## Blender

Install `blender/mimetic_blender.zip` (Edit › Preferences › Add-ons ›
Install). The **Mimetic** tab in the 3D View sidebar generates directly:

* **Mimetic head**: imports the head with a PC2 mesh cache, a jaw empty
  driving lower teeth and tongue, eyes with irises and keyframed gaze, and
  the voice in the sequencer.
* **Selected mesh (ARKit shape keys)**: keyframes your character's
  ARKit-named shape keys (VRoid, Ready Player Me, CC4 / iClone exports, and
  any Perfect Sync model). Names match case-insensitively.

The add-on runs Mimetic through the project's own Python environment, so
Blender's Python needs no extra packages.

## ARKit channel quality

The ARKit weights come from a map learned on real faces wherever it
predicts held-out speakers well (R² ≥ 0.3). That covers blink, smile, jaw
open, pucker, eye squint, inner brow raise, lower-lip depressor and funnel.
Other channels come from a FACS mapping applied directly to the muscles:

* nose sneer ← levator labii alaeque nasi
* frown ← depressor anguli oris
* dimple ← buccinator

`tongueOut` follows the articulatory tongue: it peaks on "th"/"dh", where the
tip sits between the teeth. `cheekPuff` stays at 0, since the model has no
air pressure.
