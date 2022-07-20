import torch
import torch.nn as nn


class Parameterization(nn.Module):
    def __init__(self, height, width, tensor, ema=False, decay=0.99):
        super().__init__()
        self.h, self.w = height, width

        self.tensor = nn.Parameter(tensor)

        self.ema = ema
        if ema:
            self.decay = decay
            self.register_buffer("biased", torch.zeros_like(tensor))
            self.register_buffer("average", torch.zeros_like(tensor))
            self.register_buffer("accum", torch.tensor(1.0))
            self.update_ema()

    def encode(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError()

    def decode(self, tensor: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError()

    @torch.no_grad()
    def update_ema(self):
        if self.ema:
            self.accum.mul_(self.decay)
            self.biased.mul_(self.decay)
            self.biased.add_((1 - self.decay) * self.tensor)
            self.average.copy_(self.biased)
            self.average.div_(1 - self.accum)

    @torch.no_grad()
    def reset_ema(self):
        if self.ema:
            self.biased.set_(torch.zeros_like(self.biased))
            self.average.set_(torch.zeros_like(self.average))
            self.accum.set_(torch.ones_like(self.accum))
            self.update_ema()

    def decode_average(self):
        if self.ema:
            return self.decode(self.average)
        return self.decode()


from .rgb import RGB
from .vqgan import VQGAN


def load_parameterization(which: str):
    if which == "rgb":
        return RGB
    elif which == "vqgan":
        return VQGAN
    else:
        raise Exception(f"Parameterization {which} not recognized!")
