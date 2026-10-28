"""The deployable face: activations -> skin vertices (surrogate of the simulator + data calibration)."""
from __future__ import annotations

import torch
import torch.nn as nn

from mimetic.face import anatomy as A
from mimetic.face.surrogate import MuscleSurrogate
from mimetic.paths import CHECKPOINTS


class FaceModel(nn.Module):
    def __init__(self, surrogate: MuscleSurrogate, calibration_field: torch.Tensor | None = None):
        super().__init__()
        self.sur = surrogate
        if calibration_field is None:
            calibration_field = torch.zeros(A.N_ACT, surrogate.N, 3)
        self.register_buffer("cal", calibration_field)

    @property
    def rest(self):
        return self.sur.rest

    def forward(self, a):
        x = self.sur(a)
        return x + torch.einsum("...m,mnc->...nc", a, self.cal)


def load_face_model(device="cpu", calibrated: bool = True) -> FaceModel:
    ck = torch.load(CHECKPOINTS / "surrogate.pt", map_location="cpu")
    st = ck["state"]
    sur = MuscleSurrogate(st["rest"], st["w_jaw"])
    sur.load_state_dict(st)
    field = None
    enc_p = CHECKPOINTS / "encoder.pt"
    if calibrated and enc_p.exists():
        e = torch.load(enc_p, map_location="cpu", weights_only=False)
        if e.get("cal") is not None:
            field = e["cal"]["field"] * e["cal"]["mask"]
    fm = FaceModel(sur, field).to(device).eval()
    for p in fm.parameters():
        p.requires_grad_(False)
    return fm
