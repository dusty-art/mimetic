"""Headless OpenGL renderer for the simulated face (moderngl, Metal-backed GL 4.1 on macOS).

Draws the skin (one level of Loop subdivision), eyeballs with irises, upper
teeth (skull) and lower teeth + tongue (mandible), a dark oral cavity, and an
optional overlay of every muscle coloured by its activation.
"""
from __future__ import annotations

import math

import moderngl
import numpy as np
import scipy.sparse as sp

from mimetic.face import canonical as C

VS = """
#version 330
uniform mat4 mvp; uniform mat4 model;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col;
out vec3 v_pos; out vec3 v_nrm; out vec3 v_col;
void main(){ vec4 w = model*vec4(in_pos,1.0); v_pos=w.xyz; v_nrm=mat3(model)*in_nrm; v_col=in_col; gl_Position=mvp*vec4(in_pos,1.0);}"""
FS = """
#version 330
uniform vec3 eye; uniform float wrap; uniform float spec; uniform float alpha;
in vec3 v_pos; in vec3 v_nrm; in vec3 v_col; out vec4 frag;
void main(){
  vec3 n=normalize(v_nrm); vec3 V=normalize(eye-v_pos); if(dot(n,V)<0.0) n=-n;
  vec3 L1=normalize(vec3(-0.45,0.55,0.75)); vec3 L2=normalize(vec3(0.7,0.1,0.6)); vec3 L3=normalize(vec3(0.0,0.3,-1.0));
  float d1=max((dot(n,L1)+wrap)/(1.0+wrap),0.0); float d2=max((dot(n,L2)+wrap)/(1.0+wrap),0.0);
  float rim=pow(1.0-max(dot(n,V),0.0),3.0);
  vec3 H=normalize(L1+V); float s=pow(max(dot(n,H),0.0),40.0)*spec;
  vec3 sss=vec3(0.25,0.06,0.04)*max(1.0-d1,0.0)*wrap;
  vec3 c=v_col*(0.18+0.95*d1+0.35*d2)+sss*v_col+vec3(s)+0.15*rim*vec3(0.9,0.85,1.0);
  frag=vec4(pow(c,vec3(1.0/1.15)),alpha);}"""
VS_FLAT = """
#version 330
uniform mat4 mvp; in vec3 in_pos; in vec4 in_col; out vec4 v_col;
void main(){ v_col=in_col; gl_Position=mvp*vec4(in_pos,1.0);}"""
FS_FLAT = """
#version 330
in vec4 v_col; out vec4 frag; void main(){ frag=v_col; }"""

SKIN = np.array([0.86, 0.64, 0.53])
LIP = np.array([0.72, 0.36, 0.36])


def perspective(fovy, aspect, near, far):
    f = 1 / math.tan(math.radians(fovy) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]], np.float32)


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda v: np.asarray(v, np.float32), (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def loop_subdivision(faces: np.ndarray, n_verts: int):
    """Sparse matrix S (n_new, n_verts) and new faces for one Loop subdivision step."""
    edges = {}
    def eid(a, b):
        k = (min(a, b), max(a, b))
        if k not in edges:
            edges[k] = n_verts + len(edges)
        return edges[k]
    new_faces = []
    opp = {}
    for a, b, c in faces:
        ab, bc, ca = eid(a, b), eid(b, c), eid(c, a)
        new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        for (p, q, r) in ((a, b, c), (b, c, a), (c, a, b)):
            opp.setdefault((min(p, q), max(p, q)), []).append(r)
    nb = [set() for _ in range(n_verts)]
    boundary_nb = [set() for _ in range(n_verts)]
    for (a, b), o in opp.items():
        nb[a].add(b); nb[b].add(a)
        if len(o) == 1:
            boundary_nb[a].add(b); boundary_nb[b].add(a)
    rows, cols, vals = [], [], []
    for v in range(n_verts):
        if boundary_nb[v]:
            rows += [v]; cols += [v]; vals += [0.75]
            for u in boundary_nb[v]:
                rows += [v]; cols += [u]; vals += [0.125 / max(1, len(boundary_nb[v]) / 2)]
        else:
            n = len(nb[v])
            beta = 3 / 16 if n == 3 else 3 / (8 * n)
            rows += [v]; cols += [v]; vals += [1 - n * beta]
            for u in nb[v]:
                rows += [v]; cols += [u]; vals += [beta]
    for (a, b), e in edges.items():
        o = opp[(a, b)]
        if len(o) == 2:
            rows += [e] * 4; cols += [a, b, o[0], o[1]]; vals += [3 / 8, 3 / 8, 1 / 8, 1 / 8]
        else:
            rows += [e] * 2; cols += [a, b]; vals += [0.5, 0.5]
    S = sp.csr_matrix((vals, (rows, cols)), shape=(n_verts + len(edges), n_verts))
    return S, np.array(new_faces, np.int64)


def _normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def uv_sphere(r=1.0, nu=32, nv=16):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, nu + 1), np.linspace(0, np.pi, nv + 1))
    p = np.stack([np.sin(v) * np.cos(u), np.cos(v), np.sin(v) * np.sin(u)], -1).reshape(-1, 3)
    f = []
    for i in range(nv):
        for j in range(nu):
            a = i * (nu + 1) + j; b = a + nu + 1
            f += [[a, b, a + 1], [a + 1, b, b + 1]]
    return p * r, np.array(f)


def cavity_bowl(nu=28, nv=14):
    """Back half of an ellipsoid: the dark of the mouth, open toward the lips so it never hides the tongue."""
    v, f = uv_sphere(1.0, nu, nv)
    keep = v[:, 2] <= 0.15
    remap = -np.ones(len(v), int)
    remap[keep] = np.arange(keep.sum())
    ff = np.array([[remap[a], remap[b], remap[c]] for a, b, c in f if keep[a] and keep[b] and keep[c]])
    return v[keep] * np.array([2.3, 1.7, 2.0]) + np.array([0.0, -4.6, 2.9]), ff


def teeth_arch(y_top, y_bot, z_front, width=2.15, depth=1.7, thick=0.32, n=24):
    s = np.linspace(-1, 1, n)
    x = width * s
    z = z_front - depth * s ** 2
    pts, faces = [], []
    # front face, back face strips
    for zoff in (0.0, -thick):
        for yy in (y_top, y_bot):
            for i in range(n):
                pts.append([x[i] * (1 - 0.1 * (zoff != 0)), yy, z[i] + zoff])
    P = np.array(pts)
    def idx(layer, row, i): return (layer * 2 + row) * n + i
    for layer in (0, 1):
        for i in range(n - 1):
            a, b, c, d = idx(layer, 0, i), idx(layer, 0, i + 1), idx(layer, 1, i), idx(layer, 1, i + 1)
            faces += [[a, c, b], [b, c, d]]
    for row in (0, 1):  # biting edge / gum line
        for i in range(n - 1):
            a, b, c, d = idx(0, row, i), idx(0, row, i + 1), idx(1, row, i), idx(1, row, i + 1)
            faces += [[a, b, c], [b, d, c]]
    return P, np.array(faces)


class FaceRenderer:
    def __init__(self, sim, size=(512, 512), subdiv=True):
        self.sim = sim
        self.W, self.H = size
        self.ctx = moderngl.create_standalone_context()
        self.fbo = self.ctx.simple_framebuffer(size, components=4)
        self.fbo_ms = self.ctx.framebuffer(self.ctx.renderbuffer(size, 4, samples=4),
                                           self.ctx.depth_renderbuffer(size, samples=4))
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.flat = self.ctx.program(vertex_shader=VS_FLAT, fragment_shader=FS_FLAT)
        face = sim.face
        faces = face.skin_faces
        if subdiv:
            self.S, self.faces = loop_subdivision(faces, sim.N)
        else:
            self.S, self.faces = sp.identity(sim.N, format="csr"), faces
        # Skin colour: lips redder, blend smoothly.
        X = sim.X.cpu().numpy()
        from mimetic.face import anatomy as A
        ym = A.mouth_line_y(face, X[:, 0])
        lip = np.zeros(sim.N)
        vermilion = (np.abs(X[:, 0]) < 2.7) & (X[:, 1] < ym + 0.75) & (X[:, 1] > ym - 1.0)
        lip[vermilion] = 1.0
        lip[C.UPPER_OUTER + C.LOWER_OUTER] = 0.55
        col = (1 - lip[:, None]) * SKIN + lip[:, None] * LIP
        self.colors = (self.S @ col).astype(np.float32)
        self.eye_mesh = uv_sphere(1.0, 40, 20)
        self.cavity = cavity_bowl()
        self.tongue = uv_sphere(1.0, 24, 12)
        self.set_camera()

    def set_camera(self, dist=46.0, target=(0.0, -0.6, 2.0), fov=21.0, yaw=0.0, pitch=0.0):
        ex = target[0] + dist * math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
        ey = target[1] + dist * math.sin(math.radians(pitch))
        ez = target[2] + dist * math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
        self.eye = np.array([ex, ey, ez], np.float32)
        self.view = look_at(self.eye, target)
        self.proj = perspective(fov, self.W / self.H, 1.0, 200.0)

    def _draw(self, verts, faces, colors, model=np.eye(4, dtype=np.float32), wrap=0.5, spec=0.25, alpha=1.0):
        v = verts.astype(np.float32)
        n = _normals(v, faces).astype(np.float32)
        if colors.ndim == 1:
            colors = np.tile(colors[None], (len(v), 1))
        data = np.hstack([v, n, colors.astype(np.float32)]).astype("f4")
        vbo = self.ctx.buffer(data.tobytes())
        ibo = self.ctx.buffer(faces.astype("i4").tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 3f", "in_pos", "in_nrm", "in_col")], ibo)
        mvp = self.proj @ self.view @ model
        self.prog["mvp"].write(mvp.T.copy().tobytes())
        self.prog["model"].write(model.T.copy().tobytes())
        self.prog["eye"].value = tuple(self.eye)
        self.prog["wrap"].value = wrap
        self.prog["spec"].value = spec
        self.prog["alpha"].value = alpha
        vao.render()
        vao.release(); vbo.release(); ibo.release()

    def render(self, x: np.ndarray, act: np.ndarray | None = None, jaw=None, head=None,
               gaze=(0.0, 0.0), show_muscles=False, skin_alpha=1.0, bg=(0.93, 0.93, 0.95), tongue=None) -> np.ndarray:
        """x (N,3) simulated vertices; act (N_ACT,) activations; jaw (R,t) from the sim."""
        sim = self.sim
        model = np.eye(4, dtype=np.float32) if head is None else head.astype(np.float32)
        self.fbo_ms.use()
        self.ctx.clear(*bg, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        if jaw is None:
            R, t = np.eye(3), np.zeros(3)
        else:
            R, t = jaw
        jawm = np.eye(4, dtype=np.float32); jawm[:3, :3] = R; jawm[:3, 3] = t
        # Oral cavity (dark), teeth, tongue, eyes
        cav, cf = self.cavity
        self._draw(cav, cf, np.array([0.16, 0.04, 0.05]), model, wrap=0.0, spec=0.0)
        tv, tf = teeth_arch(-3.55, -4.18, 4.85)
        self._draw(tv, tf, np.array([0.93, 0.91, 0.85]), model, spec=0.5)
        lv, lf = teeth_arch(-4.38, -5.0, 4.6, width=1.95, depth=1.6)
        self._draw(lv, lf, np.array([0.9, 0.88, 0.82]), model @ jawm, spec=0.5)
        from mimetic.face.tongue import tongue_mesh
        tv_, tf_ = tongue_mesh(np.zeros(5) if tongue is None else tongue)
        self._draw(tv_, tf_, np.array([0.72, 0.33, 0.34]), model @ jawm, spec=0.4)
        ev, ef = self.eye_mesh
        for (c, r) in sim.eyes:
            gx, gy = gaze
            d = np.array([math.sin(gx), math.sin(gy), math.cos(gx) * math.cos(gy)])
            cosang = ev @ d
            col = np.where(cosang[:, None] > 0.93, np.array([0.06, 0.04, 0.03]),
                           np.where(cosang[:, None] > 0.80, np.array([0.30, 0.20, 0.12]), np.array([0.95, 0.94, 0.92])))
            self._draw(ev * r + c, ef, col, model, wrap=0.1, spec=0.9)
        # Skin
        xs = self.S @ x
        self._draw(xs, self.faces, self.colors, model, alpha=skin_alpha)
        if show_muscles and act is not None:
            self._draw_muscles(x, act, model, R, t)
        self.ctx.copy_framebuffer(self.fbo, self.fbo_ms)
        img = np.frombuffer(self.fbo.read(components=3), np.uint8).reshape(self.H, self.W, 3)
        return img[::-1].copy()

    def _draw_muscles(self, x, act, model, R, t):
        sim = self.sim
        segs, cols = [], []
        for m, (bm) in enumerate(sim.muscles):
            o = bm.origin
            if bm.spec.origin_frame == "jaw":
                o = R @ o + t
            w = bm.weights
            if w.sum() == 0:
                continue
            ins = (w[:, None] * x).sum(0) / w.sum()
            if bm.spec.kind in ("sphincter", "marginal"):
                o = ins + (o - ins) * 0.35
            # Speech uses 10-40% of a muscle's range, so the colour scale saturates at 50%.
            a = min(1.0, float(act[m]) / 0.5) ** 0.7
            col = np.array([0.15 + 0.85 * a, 0.25 * (1 - a), 0.9 * (1 - a), 0.35 + 0.65 * a])
            segs.append((o, ins)); cols.append(col)
        mvp = self.proj @ self.view @ model
        pts, cs = [], []
        for (a, b), c in zip(segs, cols):
            # Camera-facing ribbon
            d = b - a
            side = np.cross(d, self.eye - (a + b) / 2)
            side = side / (np.linalg.norm(side) + 1e-9) * 0.11
            a2, b2 = a + np.array([0, 0, 0.9]), b + np.array([0, 0, 0.9])
            quad = [a2 - side, a2 + side, b2 + side, a2 - side, b2 + side, b2 - side]
            pts += quad; cs += [c] * 6
        data = np.hstack([np.array(pts), np.array(cs)]).astype("f4")
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(self.flat, [(vbo, "3f 4f", "in_pos", "in_col")])
        self.flat["mvp"].write(mvp.T.copy().tobytes())
        self.ctx.disable(moderngl.DEPTH_TEST)
        vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        vao.release(); vbo.release()
