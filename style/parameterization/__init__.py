import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor


class Parameterization(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.image_shape = (width, height)
        self.w = width
        self.h = height

    def encode_pil(self, pil_image: PIL.Image.Image) -> torch.Tensor:
        return self.encode(to_tensor(pil_image).permute(1, 2, 0).unsqueeze(0))

    def decode_pil(self) -> PIL.Image.Image:
        raise PIL.Image.fromarray(self.decode().squeeze(0).mul(255).round().detach().cpu().numpy().astype(np.uint8))

    def encode(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError()

    def decode(self) -> torch.Tensor:
        raise NotImplementedError()


class EMA(Parameterization):
    def __init__(self, width, height, tensor, decay):
        super().__init__(width, height)
        self.decay = decay
        self.tensor = nn.Parameter(tensor)
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

    def decode_average_pil(self):
        return self.decode_pil(self.average)
