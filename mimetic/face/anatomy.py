"""Muscles of facial expression, mastication and the eyelids, placed on the canonical mesh.

Every muscle is specified the way an anatomy atlas describes it: a bony origin,
a soft-tissue insertion, a fibre direction implied by the two, and the facial
action (FACS action unit) it produces. Right-side muscles are written out and
mirrored to the left. Coordinates are canonical centimetres
(+x = subject's left, +y = up, +z = forward).

Origins on the face are given as (x, y, depth): the skin point nearest to
(x, y) pushed `depth` cm under the skin along the inward normal. Origins off
the mesh (galea aponeurotica above the forehead, the clavicle for platysma) are
given as absolute 3D points.

`excursion` is the largest skin displacement (cm) the muscle produces at full
activation, taken from facial anthropometry of maximal voluntary expressions.
The simulator calibrates each muscle's force so it reaches that excursion.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from mimetic.face import canonical as C


@dataclass(frozen=True)
class MuscleSpec:
    name: str
    kind: str                      # linear | sphincter | marginal | lid_close | lid_open
    side: str                      # R | L | C
    origin: tuple                  # (x, y, depth) on face, or absolute xyz if origin_abs
    insertions: tuple              # ((x, y, z), ...) insertion centres (snapped to skin)
    sigma: float                   # insertion falloff (cm)
    excursion: float               # max skin displacement at a=1 (cm)
    region: str = "any"            # vertex mask, see region_mask()
    origin_abs: bool = False
    origin_frame: str = "skull"    # skull | jaw
    shortening: float = 0.35       # max fibre shortening fraction (force-length)
    protrude: float = 0.0          # forward push fraction for sphincters (lip pursing)
    bulge: float = 0.0             # belly thickening (cm) at full activation
    sigma_y: float | None = None   # anisotropic falloff (vertical)
    facs: tuple = ()
    description: str = ""


def _R(**kw):
    return MuscleSpec(side="R", **kw)


# ---------------------------------------------------------------------------
# Right-side muscles (mirrored to the left below). x < 0 is the subject's right.
# ---------------------------------------------------------------------------
RIGHT = [
    # Upper face -------------------------------------------------------------
    _R(name="frontalis_medial", kind="linear", origin=(-1.5, 12.0, 2.2), origin_abs=True,
       insertions=((-1.6, 4.9, 0),), sigma=1.2, sigma_y=2.2, excursion=1.0, region="above_eyes",
       shortening=0.2, facs=(1,), description="Inner brow raiser; lifts the medial brow and wrinkles the forehead."),
    _R(name="frontalis_lateral", kind="linear", origin=(-4.6, 12.0, 1.0), origin_abs=True,
       insertions=((-4.4, 4.8, 0),), sigma=1.2, sigma_y=2.0, excursion=0.9, region="above_eyes",
       shortening=0.2, facs=(2,), description="Outer brow raiser."),
    _R(name="corrugator", kind="linear", origin=(-1.0, 4.1, 0.9),
       insertions=((-3.0, 5.0, 0),), sigma=0.9, excursion=0.6, region="above_eyes", bulge=0.12,
       facs=(4,), description="Draws the brows down and together (frown, concentration)."),
    _R(name="depressor_supercilii", kind="linear", origin=(-1.2, 2.9, 0.6),
       insertions=((-1.9, 4.6, 0),), sigma=0.7, excursion=0.4, region="above_eyes",
       facs=(4,), description="Pulls the head of the brow down."),
    _R(name="orbicularis_oculi_orbital", kind="sphincter", origin=(-2.4, 2.4, 0.8),
       insertions=((-3.1, 2.6, 0),), sigma=2.4, excursion=0.55, region="orbit_ring",
       shortening=0.25, facs=(6, 7), description="Cheek raiser and lid tightener; crow's feet, the Duchenne smile marker."),
    _R(name="orbicularis_oculi_palpebral", kind="lid_close", origin=(-3.1, 2.6, 0.8),
       insertions=((-3.1, 2.6, 0),), sigma=1.0, excursion=0.7, region="eye_R",
       facs=(43, 45, 46), description="Closes the eyelids (blink, wink, squint)."),
    _R(name="levator_palpebrae", kind="lid_open", origin=(-3.1, 3.4, 3.0),
       insertions=((-3.1, 3.0, 0),), sigma=1.0, excursion=0.25, region="eye_R",
       facs=(5,), description="Upper lid raiser (widened eyes: surprise, fear)."),
    # Mid face ---------------------------------------------------------------
    _R(name="levator_labii_alaeque_nasi", kind="linear", origin=(-0.9, 2.3, 0.4),
       insertions=((-1.25, -1.7, 0), (-0.9, -2.9, 0)), sigma=0.6, excursion=0.55, region="nose_upperlip",
       facs=(9,), description="Nose wrinkler; lifts the nostril wing and medial upper lip (disgust)."),
    _R(name="levator_labii_superioris", kind="linear", origin=(-2.4, 1.3, 0.7),
       insertions=((-1.3, -3.4, 0),), sigma=0.7, excursion=0.6, region="upper_lip",
       facs=(10,), description="Upper lip raiser; exposes the upper teeth."),
    _R(name="zygomaticus_minor", kind="linear", origin=(-3.7, 0.6, 0.8),
       insertions=((-1.9, -3.6, 0),), sigma=0.7, excursion=0.5, region="upper_lip",
       facs=(11,), description="Deepens the nasolabial fold, lifts the lateral upper lip."),
    _R(name="zygomaticus_major", kind="linear", origin=(-5.9, 0.7, 1.4),
       insertions=((-2.75, -4.35, 0),), sigma=0.85, excursion=1.25, region="mouth_corner", bulge=0.25,
       shortening=0.3, facs=(12,), description="Lip corner puller; the smile muscle."),
    _R(name="levator_anguli_oris", kind="linear", origin=(-1.9, -1.4, 1.1),
       insertions=((-2.75, -4.35, 0),), sigma=0.75, excursion=0.6, region="mouth_corner",
       facs=(13,), description="Lifts the mouth corner vertically."),
    _R(name="risorius", kind="linear", origin=(-6.7, -4.7, 1.5),
       insertions=((-2.75, -4.35, 0),), sigma=0.7, excursion=0.8, region="mouth_corner",
       facs=(20,), description="Lip stretcher; pulls the corners sideways (fear, effort)."),
    _R(name="buccinator", kind="linear", origin=(-4.6, -4.4, 2.6),
       insertions=((-2.6, -4.35, 0),), sigma=0.95, excursion=0.55, region="mouth_corner",
       facs=(14,), description="Dimpler; tightens the corners back against the teeth, flattens the cheek."),
    _R(name="nasalis_dilator", kind="linear", origin=(-2.3, -1.5, 0.5),
       insertions=((-1.35, -1.75, 0),), sigma=0.35, excursion=0.25, region="nose",
       facs=(38,), description="Flares the nostril."),
    # Lower face (origins on the mandible move with the jaw) ------------------
    _R(name="depressor_anguli_oris", kind="linear", origin=(-3.5, -8.1, 0.6), origin_frame="jaw",
       insertions=((-2.75, -4.4, 0),), sigma=0.75, excursion=0.7, region="mouth_corner",
       facs=(15,), description="Lip corner depressor (sadness, contempt)."),
    _R(name="depressor_labii_inferioris", kind="linear", origin=(-1.8, -7.7, 0.5), origin_frame="jaw",
       insertions=((-0.95, -5.05, 0),), sigma=0.7, excursion=0.6, region="lower_lip",
       facs=(16,), description="Lower lip depressor; exposes the lower teeth."),
    _R(name="mentalis", kind="linear", origin=(-0.5, -6.2, 1.0), origin_frame="jaw",
       insertions=((-0.6, -7.5, 0),), sigma=0.8, excursion=0.45, region="chin", bulge=0.2,
       facs=(17,), description="Chin raiser; pushes the lower lip up and out (pout, doubt)."),
    _R(name="platysma", kind="linear", origin=(-6.0, -17.0, -1.0), origin_abs=True,
       insertions=((-2.9, -6.0, 0),), sigma=1.4, excursion=0.45, region="lower_face",
       shortening=0.15, facs=(21,), description="Pulls the lower lip and corners down and out (horror)."),
    # Orbicularis oris: marginal (lip rolling / pressing) and peripheral (pursing).
    _R(name="orbicularis_oris_marginal_upper", kind="marginal", origin=(0.0, -4.3, 1.4),
       insertions=((-1.2, -3.8, 0),), sigma=1.3, excursion=0.35, region="upper_vermilion",
       facs=(23, 24, 28), description="Presses and rolls the upper lip inward."),
    _R(name="orbicularis_oris_marginal_lower", kind="marginal", origin=(0.0, -4.3, 1.4),
       insertions=((-1.2, -4.8, 0),), sigma=1.3, excursion=0.35, region="lower_vermilion",
       facs=(23, 24, 28), description="Presses and rolls the lower lip inward."),
    _R(name="orbicularis_oris_peripheral_upper", kind="sphincter", origin=(0.0, -4.25, 1.2),
       insertions=((-1.3, -3.6, 0),), sigma=1.6, excursion=0.75, region="upper_lip",
       shortening=0.45, protrude=0.8, facs=(18, 22), description="Purses and protrudes the upper lip (oo, w, sh)."),
    _R(name="orbicularis_oris_peripheral_lower", kind="sphincter", origin=(0.0, -4.35, 1.2),
       insertions=((-1.3, -4.9, 0),), sigma=1.6, excursion=0.75, region="lower_lip",
       shortening=0.45, protrude=0.8, facs=(18, 22), description="Purses and protrudes the lower lip."),
]

CENTER = [
    MuscleSpec(name="procerus", kind="linear", side="C", origin=(0.0, 2.0, 0.4),
               insertions=((0.0, 4.4, 0),), sigma=0.8, excursion=0.35, region="above_eyes",
               facs=(4, 9), description="Pulls the skin between the brows down (bridge wrinkles)."),
]

# Jaw actuators drive the rigid mandible; they have no skin insertion.
JAW = ["jaw_open", "jaw_close", "jaw_protrude", "jaw_lateral_R", "jaw_lateral_L"]
JAW_FACS = {"jaw_open": (26, 27), "jaw_close": (31,), "jaw_protrude": (29,), "jaw_lateral_R": (30,), "jaw_lateral_L": (30,)}
JAW_DESCRIPTIONS = {
    "jaw_open": "Digastric and lateral pterygoid (inferior heads): opens the jaw.",
    "jaw_close": "Masseter and temporalis: closes and clenches the jaw.",
    "jaw_protrude": "Both lateral pterygoids: pushes the jaw forward.",
    "jaw_lateral_R": "Left lateral pterygoid: shifts the jaw to the subject's right.",
    "jaw_lateral_L": "Right lateral pterygoid: shifts the jaw to the subject's left.",
}


def _mirror(spec: MuscleSpec) -> MuscleSpec:
    o = spec.origin
    origin = (-o[0], o[1], o[2])
    ins = tuple((-p[0], p[1], p[2]) for p in spec.insertions)
    region = spec.region.replace("eye_R", "eye_L")
    return replace(spec, side="L", origin=origin, insertions=ins, region=region)


def all_specs() -> list[MuscleSpec]:
    out = []
    for s in RIGHT:
        out.append(replace(s, name=s.name + "_R"))
        m = _mirror(s)
        out.append(replace(m, name=s.name + "_L"))
    out += CENTER
    return out


MUSCLE_SPECS = all_specs()
MUSCLE_NAMES = [s.name for s in MUSCLE_SPECS] + JAW
N_SKIN_MUSCLES = len(MUSCLE_SPECS)
N_ACT = len(MUSCLE_NAMES)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def mouth_line_y(face: C.CanonicalFace, x: np.ndarray) -> np.ndarray:
    """Height of the lip seam at horizontal position x, extended to the TMJ beyond the corners."""
    V = face.verts
    up = V[C.UPPER_INNER]
    lo = V[C.LOWER_INNER]
    mid_x = (up[:, 0] + lo[:, 0]) / 2
    mid_y = (up[:, 1] + lo[:, 1]) / 2
    order = np.argsort(mid_x)
    ax = np.abs(x)
    corner_x = abs(V[C.NAMED["mouth_corner_L"], 0])
    corner_y = V[C.NAMED["mouth_corner_L"], 1]
    y_in = np.interp(x, mid_x[order], mid_y[order])
    slope = (0.3 - corner_y) / (6.2 - corner_x)
    y_out = corner_y + slope * (ax - corner_x)
    return np.where(ax <= corner_x, y_in, y_out)


def region_mask(face: C.CanonicalFace, region: str) -> np.ndarray:
    V = face.verts
    x, y = V[:, 0], V[:, 1]
    ym = mouth_line_y(face, x)
    nose_base = -2.55
    front = face.normals[:, 2] > -0.2
    if region == "any":
        return np.ones(len(V), bool)
    if region == "above_eyes":
        return (y > 3.55) & front
    if region == "orbit_ring":
        lids = set(C.R_UPPER_LID + C.R_LOWER_LID + C.L_UPPER_LID + C.L_LOWER_LID)
        m = np.array([i not in lids for i in range(len(V))])
        return m & (y > -0.8) & (y < 5.6)
    if region in ("eye_R", "eye_L"):
        sgn = -1 if region == "eye_R" else 1
        return (np.sign(x) == sgn) & (y > 1.4) & (y < 4.2) & (np.abs(x) > 1.4) & (np.abs(x) < 5.2)
    if region == "nose":
        return (np.abs(x) < 2.1) & (y > -2.7) & (y < 0.5)
    if region == "nose_upperlip":
        return (np.abs(x) < 2.2) & (y > ym + 0.05) & (y < 0.0)
    if region == "upper_lip":
        return (np.abs(x) < 3.2) & (y > ym + 0.02) & (y < nose_base + 0.3)
    if region == "lower_lip":
        return (np.abs(x) < 3.2) & (y < ym - 0.02) & (y > -6.6)
    if region == "upper_vermilion":
        return (np.abs(x) < 2.8) & (y > ym + 0.02) & (y < ym + 0.9)
    if region == "lower_vermilion":
        return (np.abs(x) < 2.8) & (y < ym - 0.02) & (y > ym - 1.1)
    if region == "mouth_corner":
        return (y < -1.0) & (y > -6.8) & front
    if region == "chin":
        return (np.abs(x) < 3.0) & (y < -5.6) & front
    if region == "lower_face":
        return (y < ym + 0.3) & front
    raise ValueError(region)


def jaw_weights(face: C.CanonicalFace) -> np.ndarray:
    """Skinning weight of each skin vertex to the mandible (0 = skull, 1 = jaw)."""
    V = face.verts
    x, y = V[:, 0], V[:, 1]
    ym = mouth_line_y(face, x)
    ax = np.abs(x)
    corner_x = abs(V[C.NAMED["mouth_corner_L"], 0])
    band = 0.18 + 0.32 * np.clip(ax - corner_x, 0, None)
    w = 1.0 / (1.0 + np.exp(-(ym - y) / band))
    w[C.LOWER_INNER[1:-1]] = 1.0
    w[C.UPPER_INNER[1:-1]] = 0.0
    w[[C.UPPER_INNER[0], C.UPPER_INNER[-1]]] = 0.5
    # Behind the TMJ nothing follows the jaw.
    w *= np.clip((V[:, 2] + 2.0) / 2.0, 0, 1)
    return w


def attachment_stiffness(face: C.CanonicalFace) -> np.ndarray:
    """How tightly each skin vertex is tethered to the underlying bone (relative units).

    Retaining ligaments and thin skin over bone (forehead top, nasal bridge,
    temples, jawline) are stiff; lips, cheeks and lids are loose.
    """
    V = face.verts
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    ym = mouth_line_y(face, x)
    k = np.full(len(V), 0.6)
    k[y > 6.5] = 3.0                                      # upper forehead over frontal bone
    k[(y > 4.5) & (y <= 6.5)] = 0.8                       # brow/forehead slides on galea
    k[(np.abs(x) < 1.2) & (y > 0.0) & (y < 4.0)] = 3.0     # nasal bridge
    k[(np.abs(x) < 1.6) & (y > -2.4) & (y <= 0.0)] = 1.5   # nose cartilage
    k[np.abs(x) > 5.6] = 2.5                               # temples, preauricular
    k[z < 0.5] = 4.0
    lips = (np.abs(x) < 3.0) & (y < ym + 1.4) & (y > ym - 1.6)
    k[lips] = 0.08
    cheeks = (np.abs(x) > 2.2) & (np.abs(x) < 5.6) & (y < 0.5) & (y > -6.5)
    k[cheeks] = 0.3
    chin = (np.abs(x) < 3.0) & (y < -6.6)
    k[chin] = 0.9
    eyes = (np.abs(x) > 1.4) & (np.abs(x) < 5.2) & (y > 1.6) & (y < 3.9)
    k[eyes] = 0.25
    k[C.FACE_OVAL] = 6.0
    return k


def tissue_thickness(face: C.CanonicalFace) -> np.ndarray:
    V = face.verts
    x, y = V[:, 0], V[:, 1]
    ym = mouth_line_y(face, x)
    h = np.full(len(V), 0.6)
    h[(np.abs(x) > 2.0) & (np.abs(x) < 5.8) & (y < 1.0) & (y > -6.5)] = 1.8   # cheek fat pads
    h[(np.abs(x) < 3.0) & (y < ym + 1.2) & (y > ym - 1.4)] = 1.2              # lips
    h[y > 3.5] = 0.5
    h[(np.abs(x) < 1.5) & (y > -2.5) & (y < 4.0)] = 0.3
    return h


def volume_stiffness(face: C.CanonicalFace) -> np.ndarray:
    """How strongly tissue preserves volume (bulges when compressed)."""
    V = face.verts
    x, y = V[:, 0], V[:, 1]
    ym = mouth_line_y(face, x)
    k = np.full(len(V), 0.1)
    k[(np.abs(x) > 2.0) & (np.abs(x) < 5.8) & (y < 1.0) & (y > -6.5)] = 0.6
    k[(np.abs(x) < 3.0) & (y < ym + 1.2) & (y > ym - 1.4)] = 0.5
    return k


def snap_to_skin(face: C.CanonicalFace, x: float, y: float) -> int:
    V = face.verts
    front = face.normals[:, 2] > -0.3
    d = (V[:, 0] - x) ** 2 + (V[:, 1] - y) ** 2 + np.where(front, 0, 1e6)
    return int(d.argmin())


@dataclass
class BuiltMuscle:
    spec: MuscleSpec
    origin: np.ndarray            # (3,) rest-space origin
    weights: np.ndarray           # (N,) insertion weights
    belly: np.ndarray             # (N,) belly weights for bulging
    center: np.ndarray            # (3,) sphincter centre / insertion centroid


def build_muscles(face: C.CanonicalFace | None = None) -> list[BuiltMuscle]:
    face = face or C.load_canonical()
    V, Nn = face.verts, face.normals
    built = []
    for s in MUSCLE_SPECS:
        if s.origin_abs:
            origin = np.array(s.origin, float)
        else:
            i = snap_to_skin(face, s.origin[0], s.origin[1])
            origin = V[i] - s.origin[2] * Nn[i]
        mask = region_mask(face, s.region)
        w = np.zeros(len(V))
        cents = []
        for p in s.insertions:
            j = snap_to_skin(face, p[0], p[1])
            c = V[j]
            cents.append(c)
            sy = s.sigma_y or s.sigma
            d2 = ((V[:, 0] - c[0]) / s.sigma) ** 2 + ((V[:, 1] - c[1]) / sy) ** 2 + ((V[:, 2] - c[2]) / s.sigma) ** 2
            w = np.maximum(w, np.exp(-0.5 * d2))
        if s.kind in ("marginal",) or (s.kind == "sphincter" and "oris" in s.name):
            # Lip halves: fade across the midline so left/right can act independently.
            sgn = -1.0 if s.side == "R" else 1.0
            w *= 1.0 / (1.0 + np.exp(-(sgn * V[:, 0]) / 0.35))
        w[w < 0.03] = 0
        w *= mask
        center = origin if s.kind in ("sphincter", "marginal") else np.mean(cents, 0)
        # Belly: skin over the segment between origin and insertion centroid.
        belly = np.zeros(len(V))
        if s.bulge > 0:
            a, b = origin, np.mean(cents, 0)
            ab = b - a
            t = np.clip(((V - a) @ ab) / (ab @ ab), 0.15, 0.85)
            proj = a + t[:, None] * ab
            d = np.linalg.norm((V - proj)[:, :2], axis=1)
            belly = np.exp(-0.5 * (d / 0.8) ** 2) * (Nn[:, 2] > 0)
            belly[belly < 0.05] = 0
        built.append(BuiltMuscle(s, origin, w, belly, center))
    return built
