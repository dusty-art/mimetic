"""Quasi-static muscle-driven soft-tissue face simulator (projective dynamics).

The skin is a membrane on the canonical mesh. Each vertex is held by
  * stretch springs along mesh edges (skin elasticity, rotation invariant),
  * smoothness of its deviation from the skinned bone pose (bending),
  * a tether to the bone underneath (retaining ligaments / fascia),
  * a thickness constraint to the bone that thickens when tissue is
    compressed tangentially (volume conservation -> cheek and lip bulging),
  * contact: lips cannot cross, lids cannot cross, lids slide on the eyeball,
and is pulled by muscle forces that follow a force-length curve (a muscle loses
force as it shortens, so expressions saturate the way real faces do).

Every term is quadratic in positions with constant weights, so the global
system matrix is fixed and inverted once (468 x 468, shared by x/y/z).
A solve is ~60 local/global iterations, batched over any number of poses.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from mimetic.face import anatomy as A
from mimetic.face import canonical as C

def default_rest():
    """Population mean neutral face if it has been measured, else the MediaPipe canonical mesh."""
    from mimetic.paths import ASSETS
    p = ASSETS / "rest_neutral.npy"
    return np.load(p) if p.exists() else None


KIND = {"linear": 0, "sphincter": 1, "marginal": 2, "lid_close": 3, "lid_open": 4}


def _rot_x(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([torch.stack([o, z, z], -1), torch.stack([z, c, -s], -1), torch.stack([z, s, c], -1)], -2)


def _rot_y(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1), torch.stack([-s, z, c], -1)], -2)


def jaw_rt(jaw_act: torch.Tensor, tmj=(0.0, 0.3, -1.6)):
    """Mandible pose from jaw actuators (B,5) = open, close, protrude, lateral_R, lateral_L.

    Opening is a rotation about the TMJ axis coupled with forward condylar
    translation, as in real jaw opening. Returns R (B,3,3), t (B,3).
    """
    a_open, a_close, a_prot, a_lr, a_ll = jaw_act.unbind(-1)
    theta = 0.50 * a_open - 0.035 * a_close
    yaw = 0.06 * (a_ll - a_lr)
    R = _rot_y(yaw) @ _rot_x(theta)
    P = torch.tensor(tmj, dtype=jaw_act.dtype, device=jaw_act.device)
    shift = torch.stack([0.55 * (a_ll - a_lr), -0.05 * theta, 0.55 * a_prot + 0.35 * theta], -1)
    t = P - (R @ P) + shift
    return R, t


class FaceSim:
    TMJ = (0.0, 0.3, -1.6)

    def __init__(self, rest: np.ndarray | None = None, device="cpu", dtype=torch.float32,
                 iters: int = 60, calibrate: bool = True):
        base = C.load_canonical()
        if rest is None:
            rest = default_rest()
        X = np.asarray(rest if rest is not None else base.verts, np.float64)
        # Anatomy (masks, origins, jaw weights) is laid out on the rest geometry itself.
        self.face = C.CanonicalFace(X, base.uvs, base.faces)
        self.device, self.dtype, self.iters = device, dtype, iters
        self.rho = 0.0   # Chebyshev acceleration off: contacts make it unstable
        self.N = len(X)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        ti = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
        self.X = t(X)
        faces = self.face.skin_faces
        self.faces = ti(faces)
        n0 = C.vertex_normals(X, self.face.faces)
        self.n0 = t(n0)

        # Stretch springs
        E = self.face.edges
        self.e0, self.e1 = ti(E[:, 0]), ti(E[:, 1])
        self.l0 = t(np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1))
        self.k_s = 1.0
        Lg = np.zeros((self.N, self.N))
        for a, b in E:
            Lg[a, a] += 1; Lg[b, b] += 1; Lg[a, b] -= 1; Lg[b, a] -= 1

        # Bending: uniform Laplacian of deviation from skinned rest pose
        deg = np.zeros(self.N)
        Adj = np.zeros((self.N, self.N))
        for a, b in E:
            Adj[a, b] = Adj[b, a] = 1
        deg = Adj.sum(1)
        Lu = np.eye(self.N) - Adj / deg[:, None]
        self.k_b = 2.0
        LtL = Lu.T @ Lu
        self.LtL = t(LtL)

        self.k_a = t(A.attachment_stiffness(self.face))
        self.w_jaw = t(A.jaw_weights(self.face))
        self.h = t(A.tissue_thickness(self.face))
        self.k_v = t(A.volume_stiffness(self.face))
        self.A0 = self._vertex_area(self.X[None])[0]

        # Contacts: lip seam pairs and eyelid pairs (upper index, lower index, min gap)
        lip_pairs = [(u, l) for u, l in zip(C.UPPER_INNER[1:-1], C.LOWER_INNER[1:-1])]
        lid_pairs = [(u, l) for u, l in zip(C.R_UPPER_LID[1:-1], C.R_LOWER_LID[1:-1])] + \
                    [(u, l) for u, l in zip(C.L_UPPER_LID[1:-1], C.L_LOWER_LID[1:-1])]
        pairs = lip_pairs + lid_pairs
        self.cu, self.cl = ti([p[0] for p in pairs]), ti([p[1] for p in pairs])
        self.c_gap = t([0.0] * len(lip_pairs) + [0.015] * len(lid_pairs))
        self.w_c = 20.0
        Lc = np.zeros((self.N, self.N))
        for u, l in pairs:
            Lc[u, u] += 1; Lc[l, l] += 1; Lc[u, l] -= 1; Lc[l, u] -= 1

        # Eyeballs: lids and periorbital skin slide over a sphere.
        self.eyes = []
        eye_mask = np.zeros(self.N)
        for up, lo, region in ((C.R_UPPER_LID, C.R_LOWER_LID, "eye_R"), (C.L_UPPER_LID, C.L_LOWER_LID, "eye_L")):
            rim = X[up + lo]
            centre = rim.mean(0) - np.array([0, 0, 1.05])
            radius = np.linalg.norm(rim - centre, axis=1).mean() - 0.02
            self.eyes.append((centre, radius))
            eye_mask[A.region_mask(self.face, region)] = 1.0
        self.eye_c = t(np.stack([e[0] for e in self.eyes]))
        self.eye_r = t([e[1] for e in self.eyes])
        self.eye_side = t(np.where(X[:, 0] < 0, 0, 1)).long()
        self.w_eye = 8.0
        self.eye_mask = t(eye_mask)

        Amat = (self.k_s * Lg + self.k_b * LtL + np.diag(A.attachment_stiffness(self.face))
                + np.diag(A.volume_stiffness(self.face)) + self.w_c * Lc + self.w_eye * np.diag(eye_mask))
        self.Ainv = t(np.linalg.inv(Amat))

        # Eye-line height (for lid forces) and mouth seam height (for marginal OO)
        self.mouth_y = t(A.mouth_line_y(self.face, X[:, 0]))
        eye_y = np.zeros(self.N)
        for up, lo in ((C.R_UPPER_LID, C.R_LOWER_LID), (C.L_UPPER_LID, C.L_LOWER_LID)):
            mx = (X[up, 0] + X[lo, 0]) / 2
            my = (X[up, 1] + X[lo, 1]) / 2
            o = np.argsort(mx)
            side = np.sign(mx.mean())
            sel = np.sign(X[:, 0]) == side
            eye_y[sel] = np.interp(X[sel, 0], mx[o], my[o])
        self.eye_y = t(eye_y)

        # Muscles
        self.muscles = A.build_muscles(self.face)
        M = len(self.muscles)
        self.M = M
        self.W = t(np.stack([m.weights for m in self.muscles]))           # (M, N)
        self.belly = t(np.stack([m.belly for m in self.muscles]))
        self.origin = t(np.stack([m.origin for m in self.muscles]))
        self.centre = t(np.stack([m.center for m in self.muscles]))
        self.kind = ti([KIND[m.spec.kind] for m in self.muscles])
        self.on_jaw = t([1.0 if m.spec.origin_frame == "jaw" else 0.0 for m in self.muscles])
        self.shorten = t([m.spec.shortening for m in self.muscles])
        self.protrude = t([m.spec.protrude for m in self.muscles])
        self.bulge = t([m.spec.bulge for m in self.muscles])
        self.excursion = t([m.spec.excursion for m in self.muscles])
        self.F = torch.full((M,), 1.0, dtype=dtype, device=device)
        # Sparse (muscle, vertex) insertion pairs.
        Wn = self.W.cpu().numpy()
        pm, pv = np.nonzero(Wn)
        self.pm, self.pv = ti(pm), ti(pv)
        self.pw = t(Wn[pm, pv])
        self.pbelly_m, self.pbelly_v = [ti(a) for a in np.nonzero(self.belly.cpu().numpy())]
        self.pbelly_w = self.belly[self.pbelly_m, self.pbelly_v]
        self.pkind = self.kind[self.pm]
        # Marginal orbicularis pulls each lip toward the seam; the direction is fixed at rest so a lip
        # that reaches the seam does not flip and chatter.
        self.pseam = torch.sign(self.mouth_y[self.pv] - self.X[self.pv, 1])
        # Rest lengths: linear muscles measure vertex-origin distance, sphincters the radius in the xy plane.
        d = self.X[self.pv] - self.origin[self.pm]
        self.pL0 = torch.where(self.pkind == 1, d[:, :2].norm(dim=-1), d.norm(dim=-1)).clamp_min(1e-3)
        if calibrate:
            self.calibrate()

    # ------------------------------------------------------------------
    def _vertex_area(self, x):
        f = self.faces
        a = torch.cross(x[:, f[:, 1]] - x[:, f[:, 0]], x[:, f[:, 2]] - x[:, f[:, 0]], dim=-1).norm(dim=-1) / 6
        out = torch.zeros(x.shape[0], self.N, dtype=x.dtype, device=x.device)
        for k in range(3):
            out.index_add_(1, f[:, k], a)
        return out.clamp_min(1e-6)

    def _normals(self, x):
        f = self.faces
        fn = torch.cross(x[:, f[:, 1]] - x[:, f[:, 0]], x[:, f[:, 2]] - x[:, f[:, 0]], dim=-1)
        n = torch.zeros_like(x)
        for k in range(3):
            n.index_add_(1, f[:, k], fn)
        return n / (n.norm(dim=-1, keepdim=True) + 1e-9)

    def jaw_transform(self, jaw_act: torch.Tensor):
        return jaw_rt(jaw_act, self.TMJ)

    def skinned_rest(self, jaw_act):
        R, t = self.jaw_transform(jaw_act)
        Xj = torch.einsum("bij,nj->bni", R, self.X) + t[:, None]
        w = self.w_jaw[None, :, None]
        return (1 - w) * self.X[None] + w * Xj, R, t

    # ------------------------------------------------------------------
    def muscle_forces(self, x, act, origins, T=None):
        """x (B,N,3), act (B,M), origins (B,M,3) -> force (B,N,3). Sparse over insertion pairs."""
        pm, pv, kind = self.pm, self.pv, self.pkind
        xv = x[:, pv]                                                          # (B,K,3)
        d = origins[:, pm] - xv                                                # toward origin
        planar = torch.cat([d[..., :2], torch.zeros_like(d[..., 2:])], -1)
        sph = (kind == 1)[None, :, None]
        dvec = torch.where(sph, planar, d)
        dist = dvec.norm(dim=-1).clamp_min(1e-4)
        u = dvec / dist[..., None]
        short = ((self.pL0[None] - dist) / self.pL0[None]).clamp_min(0)
        fl = (1 - (short / self.shorten[pm][None]) ** 2).clamp_min(0)
        base = (self.F[None, :] * act)[:, pm] * self.pw[None]                # (B,K)
        mag = base * torch.where(kind[None] <= 1, fl, torch.ones_like(fl))
        # Sphincter pursing: compressed lips extrude forward.
        fz = mag * self.protrude[pm][None] * (kind == 1)[None]
        # Marginal orbicularis: toward the seam and rolled back.
        seam = self.pseam[None].expand_as(xv[..., 1])
        u_m = torch.stack([torch.zeros_like(seam), seam, -0.6 * torch.ones_like(seam)], -1) / math.sqrt(1.36)
        # Eyelids: closer pulls the upper lid down (lower lid up a little); opener lifts and folds back.
        above = (xv[..., 1] > self.eye_y[pv][None]).to(x.dtype)
        u_c = torch.stack([torch.zeros_like(above), -above + 0.3 * (1 - above), torch.zeros_like(above)], -1)
        u_o = torch.stack([torch.zeros_like(above), above, -0.5 * above], -1)
        k = kind[None, :, None]
        u = torch.where(k == 2, u_m, torch.where(k == 3, u_c, torch.where(k == 4, u_o, u)))
        if T is not None:
            # Muscles without a length-based limit saturate by how far they have moved the skin.
            along = ((xv - T[:, pv]) * u).sum(-1)
            lim = 1.25 * self.excursion[pm][None]
            sat = (1 - (along.clamp_min(0) / lim) ** 2).clamp_min(0)
            mag = torch.where(kind[None] >= 2, mag * sat, mag)
        f = mag[..., None] * u
        f[..., 2] += fz
        out = torch.zeros_like(x)
        out.index_add_(1, pv, f)
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def solve(self, act: torch.Tensor, iters: int | None = None, x0=None, return_info=False):
        """act (B, N_ACT) in [0,1] (skin muscles then 5 jaw actuators) -> vertices (B, N, 3)."""
        act = act.to(self.dtype)
        B = act.shape[0]
        a_m, a_j = act[:, : self.M], act[:, self.M:]
        T, R, t = self.skinned_rest(a_j)
        jaw_origin = torch.einsum("bij,mj->bmi", R, self.origin) + t[:, None]
        origins = torch.where(self.on_jaw[None, :, None] > 0, jaw_origin, self.origin[None])
        # Bone points under each vertex (for thickness), skinned like the skin.
        Bone0 = self.X - self.h[:, None] * self.n0
        Bj = torch.einsum("bij,nj->bni", R, Bone0) + t[:, None]
        w = self.w_jaw[None, :, None]
        Bone = (1 - w) * Bone0[None] + w * Bj
        h_extra = torch.zeros(B, self.N, dtype=self.dtype, device=self.device)
        h_extra.index_add_(1, self.pbelly_v, (a_m * self.bulge[None])[:, self.pbelly_m] * self.pbelly_w[None])

        rhs_const = self.k_b * torch.einsum("nm,bmc->bnc", self.LtL, T) + self.k_a[None, :, None] * T
        x = T.clone() if x0 is None else x0.clone()
        x_prev = x.clone()
        k_it, omega = 0, 1.0
        e0, e1 = self.e0, self.e1
        for _ in range(iters or self.iters):
            rhs = rhs_const.clone()
            # stretch
            d = x[:, e0] - x[:, e1]
            p = d / (d.norm(dim=-1, keepdim=True) + 1e-9) * self.l0[None, :, None] * self.k_s
            rhs.index_add_(1, e0, p)
            rhs.index_add_(1, e1, -p)
            # thickness / volume
            n = self._normals(x)
            ratio = (self.A0[None] / self._vertex_area(x)).clamp(0.6, 1.8)
            h_t = self.h[None] * ratio + h_extra
            cur = ((x - Bone) * n).sum(-1)
            pv = x + n * (h_t - cur)[..., None]
            rhs += self.k_v[None, :, None] * pv
            # contacts (inactive -> project to current, which exerts no force)
            du = x[:, self.cu] - x[:, self.cl]
            gap = du[..., 1]
            proj = du.clone()
            proj[..., 1] = torch.maximum(gap, self.c_gap[None])
            pc = self.w_c * proj
            rhs.index_add_(1, self.cu, pc)
            rhs.index_add_(1, self.cl, -pc)
            # eyeball
            ec = self.eye_c[self.eye_side]                                    # (N,3)
            er = self.eye_r[self.eye_side]
            de = x - ec[None]
            dn = de.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            pe = torch.where(dn < er[None, :, None] + 0.04, ec[None] + de / dn * (er[None, :, None] + 0.04), x)
            rhs += self.w_eye * self.eye_mask[None, :, None] * pe
            # muscles
            rhs += self.muscle_forces(x, a_m, origins, T)
            x_hat = torch.einsum("nm,bmc->bnc", self.Ainv, rhs)
            # Chebyshev semi-iterative acceleration (Wang 2015) after a short warm-up.
            k_it += 1
            if k_it < 10 or self.rho <= 0:
                omega = 1.0
            elif k_it == 10:
                omega = 2.0 / (2.0 - self.rho ** 2)
            else:
                omega = 4.0 / (4.0 - self.rho ** 2 * omega)
            x_hat = 0.9 * x_hat + 0.1 * x
            x_new = omega * (x_hat - x_prev) + x_prev if omega != 1.0 else x_hat
            x_prev, x = x, x_new
        if return_info:
            return x, {"T": T, "R": R, "t": t}
        return x

    @torch.no_grad()
    def calibrate(self, rounds: int = 8):
        """Scale each muscle's force so full activation reaches its anatomical excursion."""
        M = self.M
        act = torch.zeros(M + 1, M + len(A.JAW), dtype=self.dtype, device=self.device)
        act[torch.arange(M), torch.arange(M)] = 1.0
        rest = self.solve(act[M:M + 1])[0]
        for _ in range(rounds):
            x = self.solve(act[:M])
            disp = (x - rest[None]).norm(dim=-1).max(dim=1).values
            scale = (self.excursion / disp.clamp_min(1e-4)).clamp(0.2, 5.0) ** 0.9
            self.F = self.F * scale
        self.rest_solved = rest
        self.calib_error = (disp / self.excursion - 1).abs().max().item()

    def neutral_act(self, B=1):
        return torch.zeros(B, self.M + len(A.JAW), dtype=self.dtype, device=self.device)

    def act_from_dict(self, d: dict, B=1):
        a = self.neutral_act(B)
        for k, v in d.items():
            a[:, A.MUSCLE_NAMES.index(k)] = v
        return a
