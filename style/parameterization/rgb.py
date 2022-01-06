import torch

from . import Parameterization


def to_colorspace(tensor, colorspace):
    if colorspace == "rgb":
        return tensor
    else:
        raise NotImplementedError()


class RGB(Parameterization):
    def __init__(self, height, width, tensor=None, colorspace="rgb"):
        if tensor is None:
            tensor = torch.empty(1, 3, height, width).uniform_()
        Parameterization.__init__(self, height, width, tensor)
        self.colorspace = colorspace

    def decode(self, tensor=None):
        if tensor is None:
            tensor = self.tensor
        return tensor.clamp(0, 1)

    @torch.no_grad()
    def encode(self, tensor):
        self.tensor.set_(to_colorspace(tensor.clamp(0, 1), self.colorspace).data)

    def forward(self):
        return self.decode()
