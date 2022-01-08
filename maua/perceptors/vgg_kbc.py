from functools import partial
from typing import Dict

import torch
import torch.nn as nn
from torchvision import models, transforms

from maua.perceptors import Perceptor


class KBCPerceptor(Perceptor):
    """VGG network by Katherine Crowson"""

    poolings = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d, "l2": partial(nn.LPPool2d, 2)}
    pooling_scales = {"max": 1.0, "avg": 2.0, "l2": 0.78}

    def __init__(
        self,
        content_layers=None,
        style_layers=None,
        content_strength=1,
        style_strength=1,
        pooling="max",
    ):
        if content_layers is None:
            content_layers = [22]
        if style_layers is None:
            style_layers = [1, 6, 11, 20, 29]

        super().__init__(content_strength, content_layers, style_strength, style_layers)

        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        net = models.vgg19(pretrained=True).features
        self.net = nn.Sequential(
            *list(net.children())[: max(content_layers + style_layers) + 1]  # remove unnecessary layers
        )

        self.net[0] = self._change_padding_mode(self.net[0], "replicate")  # Reduces edge artifacts.

        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.net):
            if pooling != "max" and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations changing, so rescale them
                self.net[i] = Scale(self.poolings[pooling](2), pool_scale)

        self.net.eval()
        self.net.requires_grad_(False)

        self.register_layer_hooks()

    @staticmethod
    def _change_padding_mode(conv: nn.Conv2d, padding_mode: str):
        new_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            padding_mode=padding_mode,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv


class Scale(nn.Module):
    def __init__(self, module: nn.Module, scale: float):
        super().__init__()
        self.module = module
        self.register_buffer("scale", torch.tensor(scale))

    def extra_repr(self):
        return f"(scale): {self.scale.item():g}"

    def forward(self, input: Dict[str, torch.Tensor]):
        return self.module(input) * self.scale
