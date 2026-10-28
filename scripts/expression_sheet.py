"""Composite expressions and speech shapes, front and side view."""
import sys
import torch
from PIL import Image, ImageDraw
from mimetic.face.sim import FaceSim
from mimetic.face.render import FaceRenderer

def both(name, v): return {name + '_R': v, name + '_L': v}
EXPR = {
  'smile': {**both('zygomaticus_major', 0.8), **both('orbicularis_oculi_orbital', 0.5), **both('levator_anguli_oris', 0.2), 'jaw_open': 0.05},
  'big_smile': {**both('zygomaticus_major', 1.0), **both('orbicularis_oculi_orbital', 0.8), **both('levator_labii_superioris', 0.4), **both('risorius', 0.3), 'jaw_open': 0.15},
  'sad': {**both('frontalis_medial', 0.8), **both('corrugator', 0.5), **both('depressor_anguli_oris', 0.8), **both('mentalis', 0.4)},
  'angry': {**both('corrugator', 1.0), **both('depressor_supercilii', 0.8), 'procerus': 0.8, **both('levator_palpebrae', 0.6), **both('orbicularis_oris_marginal_upper', 0.6), **both('orbicularis_oris_marginal_lower', 0.6), 'jaw_close': 1.0},
  'surprise': {**both('frontalis_medial', 1.0), **both('frontalis_lateral', 1.0), **both('levator_palpebrae', 1.0), 'jaw_open': 0.35},
  'disgust': {**both('levator_labii_alaeque_nasi', 1.0), **both('levator_labii_superioris', 0.5), 'procerus': 0.6, **both('corrugator', 0.3), **both('depressor_anguli_oris', 0.4)},
  'fear': {**both('frontalis_medial', 1.0), **both('corrugator', 0.5), **both('levator_palpebrae', 1.0), **both('risorius', 0.9), **both('platysma', 0.6), 'jaw_open': 0.15},
  'pucker_oo': {**both('orbicularis_oris_peripheral_upper', 1.0), **both('orbicularis_oris_peripheral_lower', 1.0), 'jaw_open': 0.08},
  'm_closed': {**both('orbicularis_oris_marginal_upper', 0.5), **both('orbicularis_oris_marginal_lower', 0.5), **both('mentalis', 0.3), 'jaw_close': 0.5},
  'ee': {**both('risorius', 0.6), **both('zygomaticus_major', 0.3), **both('buccinator', 0.5), 'jaw_open': 0.12},
  'aa': {'jaw_open': 0.45, **both('depressor_labii_inferioris', 0.3)},
  'f_v': {**both('orbicularis_oris_marginal_lower', 0.8), **both('levator_labii_superioris', 0.3), 'jaw_open': 0.04},
}
out = sys.argv[1] if len(sys.argv) > 1 else 'experiments/anatomy/expressions.png'
sim = FaceSim()
r = FaceRenderer(sim, (256, 256))
names = list(EXPR)
acts = torch.cat([sim.act_from_dict(EXPR[n]) for n in names])
x, info = sim.solve(acts, return_info=True)
sheet = Image.new('RGB', (len(names) // 2 * 256, 4 * 256), 'white')
for k, n in enumerate(names):
    jaw = (info['R'][k].numpy(), info['t'][k].numpy())
    for v, yaw in enumerate((0, 70)):
        r.set_camera(yaw=yaw)
        im = Image.fromarray(r.render(x[k].numpy(), jaw=jaw))
        d = ImageDraw.Draw(im); d.text((3, 1), n, fill=(0, 0, 0))
        sheet.paste(im, ((k % 6) * 256, (2 * (k // 6) + v) * 256))
sheet.save(out); print(out)
