"""3D exports: OBJ head, PC2 point caches (Blender Mesh Cache modifier, Maya, 3ds Max), jaw/eye tracks.

Blender convention: metres, Z up, -Y forward. Mimetic: centimetres, Y up, +Z forward.
"""
from __future__ import annotations

import struct

import numpy as np

from mimetic.face import canonical as C
from mimetic.face.render import loop_subdivision


def to_blender(v):
    v = np.asarray(v, np.float32) * 0.01
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1)


class SkinTopology:
    def __init__(self, subdivide=True):
        face = C.load_canonical()
        if subdivide:
            self.S, self.faces = loop_subdivision(face.skin_faces, C.N_VERTS)
        else:
            import scipy.sparse as sp
            self.S, self.faces = sp.identity(C.N_VERTS, format="csr"), face.skin_faces

    def expand(self, verts):
        """(T, 468, 3) -> (T, n_sub, 3)"""
        return np.stack([self.S @ v for v in verts]).astype(np.float32)


def write_obj(path, verts, faces, name="mimetic_head"):
    with open(path, "w") as f:
        f.write(f"o {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def write_pc2(path, frames, fps, start_frame=1.0):
    frames = np.asarray(frames, np.float32)
    with open(path, "wb") as f:
        f.write(b"POINTCACHE2\0")
        f.write(struct.pack("<iiffi", 1, frames.shape[1], float(start_frame), 1.0, frames.shape[0]))
        f.write(frames.tobytes())


def jaw_tracks(acts, tmj=(0.0, 0.3, -1.6)):
    """Per-frame 4x4 mandible transforms (Mimetic space) from the 5 jaw activations."""
    import torch
    from mimetic.face.sim import jaw_rt
    R, t = jaw_rt(torch.as_tensor(acts[:, -5:], dtype=torch.float32), tmj)
    M = np.tile(np.eye(4, dtype=np.float32), (len(acts), 1, 1))
    M[:, :3, :3] = R.numpy()
    M[:, :3, 3] = t.numpy()
    return M


def write_props(outdir):
    """Teeth, tongue, eyes and oral cavity as OBJ (Blender space); lower teeth and tongue are jaw-relative."""
    from pathlib import Path
    from mimetic.face.render import cavity_bowl, teeth_arch, uv_sphere
    from mimetic.face.sim import FaceSim
    outdir = Path(outdir)
    sim = FaceSim(calibrate=False)
    v, f = teeth_arch(-3.55, -4.18, 4.85)
    write_obj(outdir / "upper_teeth.obj", to_blender(v), f, "upper_teeth")
    v, f = teeth_arch(-4.38, -5.0, 4.6, width=1.95, depth=1.6)
    write_obj(outdir / "lower_teeth.obj", to_blender(v), f, "lower_teeth")
    sv, sf = uv_sphere(1.0, 24, 12)
    write_obj(outdir / "tongue.obj", to_blender(sv * np.array([1.5, 0.45, 1.7]) + np.array([0.0, -4.85, 3.1])), sf, "tongue")
    cv, cf = cavity_bowl()
    write_obj(outdir / "mouth_cavity.obj", to_blender(cv), cf, "mouth_cavity")
    ev, ef = uv_sphere(1.0, 96, 64)
    for (c, r), side in zip(sim.eyes, ("R", "L")):
        write_obj(outdir / f"eye_{side}.obj", to_blender(ev * r + c), ef, f"eye_{side}")
        # iris + pupil: a cap of the eyeball facing forward (+z), slightly proud of the sclera
        keep = ev[:, 2] > np.cos(np.radians(32))
        remap = -np.ones(len(ev), int)
        remap[keep] = np.arange(keep.sum())
        cf = np.array([[remap[a], remap[b], remap[c_]] for a, b, c_ in ef if keep[a] and keep[b] and keep[c_]])
        write_obj(outdir / f"iris_{side}.obj", to_blender(ev[keep] * (r + 0.01) + c), cf, f"iris_{side}")


def write_tongue_cache(outdir, anim):
    """Per-frame tongue vertices (jaw-relative, Blender space) as a PC2 cache for the tongue object."""
    from pathlib import Path
    from mimetic.face.tongue import tongue_mesh
    if anim.tongue is None:
        return
    frames = np.stack([to_blender(tongue_mesh(p)[0]) for p in anim.tongue])
    v, f = tongue_mesh(np.zeros(5))
    write_obj(Path(outdir) / "tongue.obj", to_blender(v), f, "tongue")
    write_pc2(Path(outdir) / "tongue.pc2", frames, anim.fps)
