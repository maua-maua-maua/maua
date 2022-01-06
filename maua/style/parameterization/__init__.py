import torch
import torch.nn as nn


class Parameterization(nn.Module):
    def __init__(self, height, width, tensor):
        super().__init__()
        self.w = width
        self.h = height
        self.tensor = nn.Parameter(tensor)

    def encode(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError()

    def decode(self) -> torch.Tensor:
        raise NotImplementedError()


class EMA(Parameterization):
    def __init__(self, height, width, tensor, decay):
        super().__init__(height, width, tensor)
        self.decay = decay
        self.register_buffer("biased", torch.zeros_like(tensor))
        self.register_buffer("average", torch.zeros_like(tensor))
        self.register_buffer("accum", torch.tensor(1.0))
        self.update()

    @torch.no_grad()
    def update(self):
        self.accum.mul_(self.decay)
        self.biased.mul_(self.decay)
        self.biased.add_((1 - self.decay) * self.tensor)
        self.average.copy_(self.biased)
        self.average.div_(1 - self.accum)

    @torch.no_grad()
    def reset(self):
        self.biased.set_(torch.zeros_like(self.biased))
        self.average.set_(torch.zeros_like(self.average))
        self.accum.set_(torch.ones_like(self.accum))
        self.update()

    def decode_average(self):
        return self.decode(self.average)
