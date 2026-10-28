import sys, os, bpy
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT + "/blender")
import mimetic_blender
mimetic_blender.register()
scn = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
scn.render.fps = 30
s = scn.mimetic
s.project = ROOT
s.text = "[happy:0.8] Hello from Blender! Bob bought a blue boat."
r = bpy.ops.mimetic.generate()
print("operator:", r, "frames", scn.frame_start, scn.frame_end)
print("objects:", [o.name for o in bpy.data.objects])
head = bpy.data.objects.get("MimeticHead")
print("verts", len(head.data.vertices), "modifiers", [m.type for m in head.modifiers])
# camera + light, render one frame
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); scn.collection.objects.link(cam)
cam.location = (0, -0.55, -0.005); cam.rotation_euler = (1.5708, 0, 0); cam.data.lens = 85
scn.camera = cam
light = bpy.data.objects.new("key", bpy.data.lights.new("key", "SUN")); scn.collection.objects.link(light)
light.rotation_euler = (0.9, 0.2, -0.4); light.data.energy = 3
scn.render.engine = "BLENDER_WORKBENCH" if "--fast" in sys.argv else "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
scn.render.resolution_x = scn.render.resolution_y = 384
for f in (12, 40):
    scn.frame_set(f)
    scn.render.filepath = ROOT + f"/experiments/blender/frame_{f}.png"
    bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=ROOT + "/experiments/blender/mimetic_test.blend")
print("saved")
