"""Contact sheet: each right-side muscle (and jaw actuator) at full activation."""
import sys
import numpy as np, torch
from PIL import Image, ImageDraw
from mimetic.face.sim import FaceSim
from mimetic.face.render import FaceRenderer
from mimetic.face import anatomy as A

out = sys.argv[1] if len(sys.argv) > 1 else 'experiments/anatomy/sheet.png'
level = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
sim = FaceSim()
r = FaceRenderer(sim, (256, 256))
names = [n for n in A.MUSCLE_NAMES if not n.endswith('_L')]
acts = torch.zeros(len(names), A.N_ACT)
for i, n in enumerate(names):
    acts[i, A.MUSCLE_NAMES.index(n)] = level
x, info = sim.solve(acts, return_info=True)
cols = 6
rows = (len(names) + 1 + cols - 1) // cols
sheet = Image.new('RGB', (cols * 256, rows * 256), 'white')
x0, i0 = sim.solve(sim.neutral_act(), return_info=True)
tiles = [('neutral', r.render(x0[0].numpy(), jaw=(i0['R'][0].numpy(), i0['t'][0].numpy())))]
for i, n in enumerate(names):
    tiles.append((n, r.render(x[i].numpy(), act=acts[i].numpy(), jaw=(info['R'][i].numpy(), info['t'][i].numpy()))))
for k, (n, img) in enumerate(tiles):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 256, 14], fill=(255, 255, 255))
    d.text((3, 1), n.replace('_R', ''), fill=(0, 0, 0))
    sheet.paste(im, ((k % cols) * 256, (k // cols) * 256))
sheet.save(out)
print(out, sheet.size)
