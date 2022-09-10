from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .ops.image import resample
from .utility import fetch


class TextPrompt(torch.nn.Module):
    def __init__(self, text, weight=1.0):
        super().__init__()
        self.text = text
        self.weight = weight

    def forward(self):
        return self.text, self.weight


class ImagePrompt(torch.nn.Module):
    def __init__(self, img=None, path=None, size=None, weight=1.0):
        super().__init__()
        self.weight = weight

        if path is not None:
            allowed_types = (str, Path)
            assert isinstance(path, allowed_types), f"path must be one of {allowed_types}"
            img = Image.open(fetch(path)).convert("RGB")
            img = to_tensor(img).unsqueeze(0)

        elif img is not None:
            allowed_types = (Image.Image, torch.Tensor, np.ndarray)
            assert isinstance(img, allowed_types), f"img must be one of {allowed_types}"
            if isinstance(img, (Image.Image, np.ndarray)):
                img = to_tensor(img).unsqueeze(0)
            else:
                assert img.dim() == 4, "img must be of shape (B, C, H, W)"

        else:
            raise Exception("path or img must be specified")

        if size is not None:
            img = resample(img, min(size))

        self.register_buffer("img", img.mul(2).sub(1))

    def forward(self):
        return self.img, self.weight


class StylePrompt(ImagePrompt):
    pass


class ContentPrompt(ImagePrompt):
    pass
