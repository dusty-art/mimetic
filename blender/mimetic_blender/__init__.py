"""Mimetic for Blender: generate muscle-driven talking animation from text or audio.

Install: Edit > Preferences > Add-ons > Install... > pick this folder zipped (or
symlink it into your add-ons directory). Set "Mimetic project" to the folder
that contains `.venv` and `mimetic/`.

Targets
  * Mimetic head : imports the muscle-simulated head (skin via a Mesh Cache
                   modifier on a PC2 point cache, animated jaw/teeth, eyes).
  * Selected mesh with ARKit shape keys : keyframes the 52 ARKit shape keys
                   (VRoid / Perfect Sync / Ready Player Me / CC / MetaHuman-style rigs).
The audio is added to the sequencer so playback is in sync.
"""
bl_info = {
    "name": "Mimetic talking face",
    "author": "Mimetic",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Mimetic",
    "description": "Muscle-driven speech + emotion facial animation from text or audio",
    "category": "Animation",
}

import json
import os
import subprocess
import tempfile

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty

VOICES = [(v, v, "") for v in ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael",
                               "am_fenrir", "am_puck", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]]


def _prefs(context):
    return context.scene.mimetic


class MimeticSettings(bpy.types.PropertyGroup):
    project: StringProperty(name="Mimetic project", subtype="DIR_PATH",
                            default=os.path.expanduser("~/Desktop/test"))
    source: EnumProperty(name="Source", items=[("TEXT", "Text", ""), ("AUDIO", "Audio file", "")])
    text: StringProperty(name="Text", default="[happy:0.7] Hello! I'm moving forty-seven muscles to say this.")
    audio: StringProperty(name="Audio", subtype="FILE_PATH")
    transcript: StringProperty(name="Transcript (optional)")
    voice: EnumProperty(name="Voice", items=VOICES)
    gain: FloatProperty(name="Articulation", default=1.0, min=0.5, max=1.6)
    target: EnumProperty(name="Target", items=[("HEAD", "Mimetic head", "Import the simulated head"),
                                              ("ARKIT", "Selected mesh (ARKit shape keys)", "")])
    add_audio: BoolProperty(name="Add audio to sequencer", default=True)


def run_export(s, fps, outdir):
    py = os.path.join(bpy.path.abspath(s.project), ".venv", "bin", "python")
    cmd = [py, "-m", "mimetic.cli", "export", "-o", outdir, "--fps", str(fps), "--gain", str(s.gain)]
    if s.source == "TEXT":
        cmd += ["--text", s.text, "--voice", s.voice]
    else:
        cmd += ["--audio", bpy.path.abspath(s.audio)]
        if s.transcript:
            cmd += ["--transcript", s.transcript]
    r = subprocess.run(cmd, cwd=bpy.path.abspath(s.project), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])


def import_head(context, outdir, fps):
    import numpy as np
    head = _import_obj(os.path.join(outdir, "head.obj"))
    mat = bpy.data.materials.new("MimeticSkin")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.86, 0.64, 0.53, 1)
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.2
        bsdf.inputs["Roughness"].default_value = 0.55
    head.data.materials.append(mat)
    for p in head.data.polygons:
        p.use_smooth = True
    mod = head.modifiers.new("MimeticMuscles", "MESH_CACHE")
    mod.cache_format = "PC2"
    mod.filepath = os.path.join(outdir, "head.pc2")
    mod.frame_start = 1
    mod.forward_axis = "POS_Y"
    mod.up_axis = "POS_Z"
    # Lower jaw (teeth) as an empty driven by keyframes
    jaw = np.load(os.path.join(outdir, "jaw.npy"))
    empty = bpy.data.objects.new("MimeticJaw", None)
    context.collection.objects.link(empty)
    import mathutils
    C = mathutils.Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    for f, M in enumerate(jaw):
        m = mathutils.Matrix(M.tolist())
        m.translation *= 0.01
        empty.matrix_world = C @ m @ C.inverted()
        empty.keyframe_insert("location", frame=f + 1)
        empty.keyframe_insert("rotation_euler", frame=f + 1)
    # Teeth, tongue and eyes
    for name, col, parent in (("upper_teeth", (0.93, 0.91, 0.85, 1), None), ("lower_teeth", (0.9, 0.88, 0.82, 1), empty),
                              ("tongue", (0.7, 0.32, 0.33, 1), empty), ("eye_R", (0.95, 0.94, 0.92, 1), None),
                              ("eye_L", (0.95, 0.94, 0.92, 1), None), ("mouth_cavity", (0.16, 0.04, 0.05, 1), None)):
        path = os.path.join(outdir, f"{name}.obj")
        if not os.path.exists(path):
            continue
        ob = _import_obj(path, name)
        m = bpy.data.materials.new(f"Mimetic_{name}")
        m.diffuse_color = col
        m.use_nodes = True
        b = m.node_tree.nodes.get("Principled BSDF")
        if b:
            b.inputs["Base Color"].default_value = col
        ob.data.materials.append(m)
        for p in ob.data.polygons:
            p.use_smooth = True
        if parent is not None:
            # Jaw-relative geometry is stored at the rest pose, so the child follows the jaw keys directly.
            ob.parent = parent
            ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    head.select_set(True)
    context.view_layer.objects.active = head
    return head


def _import_obj(path, name="MimeticHead"):
    # Minimal OBJ reader so the add-on works without the OBJ importer being enabled.
    verts, faces = [], []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append(tuple(float(x) for x in p[1:4]))
        elif p[0] == "f":
            faces.append(tuple(int(x.split("/")[0]) - 1 for x in p[1:]))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    return ob


def key_arkit(obj, outdir):
    rec = json.load(open(os.path.join(outdir, "animation.json")))
    names, frames = rec["arkit_names"], rec["arkit"]
    keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else {}
    lookup = {k.name.lower().replace("_", "").replace(".", ""): k for k in keys}
    matched = 0
    for j, n in enumerate(names):
        k = lookup.get(n.lower()) or lookup.get(("blendshape" + n).lower())
        if k is None:
            continue
        matched += 1
        for f, row in enumerate(frames):
            k.value = row[j]
            k.keyframe_insert("value", frame=f + 1)
    return matched


class MIMETIC_OT_generate(bpy.types.Operator):
    bl_idname = "mimetic.generate"
    bl_label = "Generate talking animation"

    def execute(self, context):
        s = _prefs(context)
        fps = context.scene.render.fps / context.scene.render.fps_base
        outdir = tempfile.mkdtemp(prefix="mimetic_")
        try:
            run_export(s, fps, outdir)
        except Exception as e:
            self.report({"ERROR"}, f"Mimetic failed: {e}")
            return {"CANCELLED"}
        rec = json.load(open(os.path.join(outdir, "animation.json")))
        n = rec["n_frames"]
        if s.target == "HEAD":
            import_head(context, outdir, fps)
        else:
            obj = context.active_object
            if obj is None or obj.type != "MESH":
                self.report({"ERROR"}, "Select a mesh with ARKit-named shape keys")
                return {"CANCELLED"}
            m = key_arkit(obj, outdir)
            self.report({"INFO"}, f"Keyed {m}/52 ARKit shape keys")
        wav = os.path.join(outdir, "audio.wav")
        if s.add_audio and os.path.exists(wav):
            scn = context.scene
            if not scn.sequence_editor:
                scn.sequence_editor_create()
            scn.sequence_editor.sequences.new_sound("Mimetic voice", wav, channel=1, frame_start=1)
        context.scene.frame_start, context.scene.frame_end = 1, n
        return {"FINISHED"}


class MIMETIC_PT_panel(bpy.types.Panel):
    bl_label = "Mimetic"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mimetic"

    def draw(self, context):
        s = _prefs(context)
        L = self.layout
        L.prop(s, "project")
        L.prop(s, "source", expand=True)
        if s.source == "TEXT":
            L.prop(s, "text")
            L.prop(s, "voice")
        else:
            L.prop(s, "audio")
            L.prop(s, "transcript")
        L.prop(s, "gain")
        L.prop(s, "target")
        L.prop(s, "add_audio")
        L.operator("mimetic.generate", icon="PLAY")


classes = (MimeticSettings, MIMETIC_OT_generate, MIMETIC_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.mimetic = bpy.props.PointerProperty(type=MimeticSettings)


def unregister():
    del bpy.types.Scene.mimetic
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
