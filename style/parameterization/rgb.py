import torch
import torch.nn as nn
from image_ops import resample
from losses import clamp_with_grad

from . import EMA, Parameterization


def to_colorspace(tensor, colorspace):
    if colorspace == "rgb":
        return tensor
    else:
        raise NotImplementedError()


class RGB(Parameterization):
    def __init__(self, width, height, scale=1, colorspace="rgb", ema=True):
        (EMA if ema else Parameterization).__init__(width * scale, height * scale)
        self.scale = scale
        self.colorspace = colorspace
        self.tensor = nn.Parameter(torch.empty(1, height, width, 3).uniform_().to(memory_format=torch.channels_last))

    def decode(self):
        out = resample(self.tensor, (self.h * self.scale, self.w * self.scale))
        return clamp_with_grad(out, 0, 1)

    def encode(self, tensor):
        self.tensor.data.set_(
            to_colorspace(resample(tensor, (self.h // self.scale, self.w // self.scale)), self.colorspace).data
        )
