"""Canonical face mesh (MediaPipe topology, 468 vertices, centimetres).

Coordinate frame: +x = subject's left, +y = up, +z = forward (out of the face).
The same topology is what MediaPipe Face Landmarker tracks in video, so every
tracked frame is a dense correspondence to this mesh.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np

from mimetic.paths import ASSETS

N_VERTS = 468

# Lip contours, ordered from the subject's right corner (x<0) to the left corner.
UPPER_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Eyelid margins, ordered outer corner -> inner corner.
R_UPPER_LID = [33, 246, 161, 160, 159, 158, 157, 173, 133]
R_LOWER_LID = [33, 7, 163, 144, 145, 153, 154, 155, 133]
L_UPPER_LID = [263, 466, 388, 387, 386, 385, 384, 398, 362]
L_LOWER_LID = [263, 249, 390, 373, 374, 380, 381, 382, 362]

FACE_OVAL = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234,
             251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454]

R_BROW = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55]
L_BROW = [300, 293, 334, 296, 336, 276, 283, 282, 295, 285]

NAMED = {
    "nose_tip": 1,
    "nose_bridge": 6,
    "glabella": 168,
    "forehead_top": 10,
    "chin": 152,
    "mouth_corner_R": 61,
    "mouth_corner_L": 291,
    "upper_lip_top": 0,
    "lower_lip_bottom": 17,
    "upper_lip_inner": 13,
    "lower_lip_inner": 14,
    "eye_outer_R": 33,
    "eye_inner_R": 133,
    "eye_outer_L": 263,
    "eye_inner_L": 362,
    "upper_lid_R": 159,
    "lower_lid_R": 145,
    "upper_lid_L": 386,
    "lower_lid_L": 374,
    "cheek_R": 50,
    "cheek_L": 280,
    "temple_R": 234,
    "temple_L": 454,
}

# Vertices on rigid bone with almost no soft tissue motion. Used for head-pose
# alignment of tracked frames and as stiff anchors in the simulator.
RIGID_ANCHORS = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 109, 67, 103, 338, 297,
                 332, 234, 454, 127, 356, 93, 323, 162, 389, 21, 251, 54, 284]


def _load_obj(path):
    verts, uvs, faces = [], [], []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append([float(x) for x in p[1:4]])
        elif p[0] == "vt":
            uvs.append([float(x) for x in p[1:3]])
        elif p[0] == "f":
            faces.append([int(x.split("/")[0]) - 1 for x in p[1:4]])
    return np.array(verts, np.float64), np.array(uvs, np.float64), np.array(faces, np.int64)


@dataclass
class CanonicalFace:
    verts: np.ndarray            # (468, 3) rest positions, cm
    uvs: np.ndarray              # (468, 2)
    faces: np.ndarray            # (F, 3) triangles, outward winding
    edges: np.ndarray = field(init=False)
    normals: np.ndarray = field(init=False)
    neighbors: list = field(init=False)

    def __post_init__(self):
        f = self.skin_faces
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        e = np.sort(e, axis=1)
        self.edges = np.unique(e, axis=0)
        self.normals = vertex_normals(self.verts, self.faces)
        nb = [[] for _ in range(len(self.verts))]
        for a, b in self.edges:
            nb[a].append(b)
            nb[b].append(a)
        self.neighbors = nb

    @functools.cached_property
    def skin_faces(self) -> np.ndarray:
        """Triangles with the mouth and eye openings cut out.

        The MediaPipe tessellation bridges the lips and eyelids with triangles;
        a moving face needs real holes there.
        """
        holes = [set(UPPER_INNER + LOWER_INNER), set(R_UPPER_LID + R_LOWER_LID),
                 set(L_UPPER_LID + L_LOWER_LID)]
        keep = [not any(all(v in h for v in t) for h in holes) for t in self.faces]
        return self.faces[np.array(keep)]

    def mirror_index(self) -> np.ndarray:
        """Index of the bilateral mirror vertex (x -> -x) for every vertex."""
        m = self.verts * np.array([-1.0, 1.0, 1.0])
        d = ((self.verts[None, :, :] - m[:, None, :]) ** 2).sum(-1)
        return d.argmin(1)


def vertex_normals(verts, faces):
    v = verts
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n


@functools.lru_cache(maxsize=1)
def load_canonical() -> CanonicalFace:
    v, uv, f = _load_obj(ASSETS / "canonical_face_model.obj")
    face = CanonicalFace(v, uv, f)
    # Make sure triangles wind outward (normals point +z on the face front).
    if face.normals[NAMED["nose_tip"], 2] < 0:
        face = CanonicalFace(v, uv, f[:, ::-1].copy())
    return face
