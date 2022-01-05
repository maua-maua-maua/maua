import torch
import torch.nn as nn
from ops.image import resample
from ops.loss import clamp_with_grad

from . import EMA, Parameterization


def to_colorspace(tensor, colorspace):
    if colorspace == "rgb":
        return tensor
    else:
        raise NotImplementedError()


class RGB(Parameterization):
    def __init__(self, height, width, tensor=None, scale=1, colorspace="rgb"):

        if tensor is None:
            tensor = torch.empty(1, 3, height, width).uniform_()

        Parameterization.__init__(self, height * scale, width * scale, tensor)

        self.scale = scale
        self.colorspace = colorspace

    def decode(self, tensor=None):
        if tensor is None:
            tensor = self.tensor
        out = resample(tensor, (self.h * self.scale, self.w * self.scale))
        return clamp_with_grad(out, 0, 1)

    @torch.no_grad()
    def encode(self, tensor):
        self.tensor.set_(
            to_colorspace(resample(tensor, (self.h // self.scale, self.w // self.scale)), self.colorspace).data
        )

    def forward(self):
        return self.decode()
