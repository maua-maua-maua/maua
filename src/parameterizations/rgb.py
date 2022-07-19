import torch

from ..ops.loss import clamp_with_grad
from . import Parameterization


def to_colorspace(tensor, colorspace):
    if colorspace == "rgb":
        return tensor
    else:
        raise NotImplementedError()


class RGB(Parameterization):
    def __init__(self, height, width, tensor=None, colorspace="rgb", ema=False):
        if tensor is None:
            tensor = torch.empty(1, 3, height, width).uniform_().mul(0.1)
        Parameterization.__init__(self, height, width, tensor, ema)
        self.colorspace = colorspace

    def decode(self, tensor=None):
        if tensor is None:
            tensor = self.tensor
        return clamp_with_grad(tensor, 0, 1)

    @torch.no_grad()
    def encode(self, tensor):
        self.tensor.set_(to_colorspace(tensor.clamp(0, 1), self.colorspace).data)

    def forward(self):
        return self.decode()
