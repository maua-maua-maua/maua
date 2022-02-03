import warnings
from typing import Tuple

import numpy as np
import torch
from torch.nn.functional import interpolate, pad

import sys
import os

sys.path.append(os.path.dirname(__file__) + "/../nv")
from ..load import load_network
from ..nv.networks import stylegan3
from . import MauaSynthesizer
from .stylegan import StyleGAN, StyleGANMapper

layer_multipliers = {
    1024: {0: 64, 1: 64, 2: 64, 3: 32, 4: 32, 5: 16, 6: 8, 7: 8, 8: 4, 9: 4, 10: 2, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1},
    512: {0: 32, 1: 32, 2: 32, 3: 16, 4: 16, 5: 8, 6: 8, 7: 4, 8: 4, 9: 2, 10: 2, 11: 1, 12: 1, 13: 1, 14: 1},
    256: {0: 16, 1: 16, 2: 16, 3: 16, 4: 8, 5: 8, 6: 4, 7: 4, 8: 2, 9: 2, 10: 2, 11: 1, 12: 1, 13: 1, 14: 1},
}


class StyleGAN3(StyleGAN):
    def __init__(self, mapper, synthesizer) -> None:
        super().__init__(mapper, synthesizer)
        self.synthesizer.avg_shift = self.synthesizer.input.affine(self.mapper.w_avg.unsqueeze(0)).squeeze(0)


class StyleGAN3Mapper(StyleGANMapper):
    MappingNetwork = stylegan3.MappingNetwork


class StyleGAN3Synthesizer(MauaSynthesizer):
    def __init__(self, model_file: str, output_size: Tuple[int, int], strategy: str, layer: int) -> None:
        super().__init__()

        if model_file is None or model_file == "None":
            self.G_synth = stylegan3.SynthesisNetwork(w_dim=512, img_resolution=1024, img_channels=3)
        else:
            self.G_synth: stylegan3.SynthesisNetwork = load_network(model_file).synthesis

        self.w_dim, self.num_ws = self.G_synth.w_dim, self.G_synth.num_ws

        self.modulation_targets = {
            "latent_w": (self.w_dim,),
            "latent_w_plus": (self.num_ws, self.w_dim),
            "translation": (2,),
            "rotation": (1,),
        }

        self.change_output_resolution(output_size, strategy, layer)

    def forward(
        self,
        latent_w: torch.Tensor = None,
        latent_w_plus: torch.Tensor = None,
        translation: torch.Tensor = None,
        rotation: torch.Tensor = None,
    ) -> torch.Tensor:
        if latent_w is None and latent_w_plus is None:
            raise Exception("One of latent_w or latent_w_plus inputs must be supplied!")
        if latent_w is not None and latent_w_plus is not None:
            warnings.warn("Both latent_w and latent_w_plus supplied, using latent_w_plus input...")

        if not (translation is None or rotation is None):
            self.G_synth.input.transform.copy_(make_transform_mat(translation, rotation))
        else:
            # stabilization trick by @RiversHaveWings and @nshepperd1
            self.G_synth.input.affine.bias.data.add_(self.avg_shift)
            self.G_synth.input.affine.weight.data.zero_()

        if latent_w_plus is None:
            latent_w_plus = torch.tile(latent_w[:, None, :], (1, self.G_synth.num_ws, 1))

        return self.G_synth.forward(latent_w_plus)

    def change_output_resolution(self, output_size: Tuple[int, int], strategy: str, layer: int):
        self.refresh_model_hooks()

        if output_size != (self.G_synth.img_resolution, self.G_synth.img_resolution):
            lay_mult = layer_multipliers[self.G_synth.img_resolution][layer]

            unrounded_size = np.array(output_size) / lay_mult + 20
            size = np.round(unrounded_size).astype(int)
            if sum(abs(unrounded_size - size)) > 1e-10:
                warnings.warn(
                    f"Layer {layer} resizes to multiples of {lay_mult}. --output-size rounded to {lay_mult * (size - 20)}"
                )

            synth_layer = getattr(self.G_synth, "input" if layer == 0 else self.G_synth.layer_names[layer - 1])
            hook = synth_layer.register_forward_hook(get_hook(self.G_synth, layer, size, strategy))
            self._hook_handles.append(hook)

        self.output_size = output_size


def make_transform_mat(translate: Tuple[float, float], angle: float) -> torch.Tensor:
    s = np.sin(angle.squeeze().cpu() / 360.0 * np.pi * 2)
    c = np.cos(angle.squeeze().cpu() / 360.0 * np.pi * 2)
    m = np.array([[c, s, translate.squeeze().cpu()[0]], [-s, c, translate.squeeze().cpu()[1]], [0, 0, 0]])
    try:
        m = np.linalg.inv(m)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"Singular transform matrix, continuing with pseudo-inverse of transform matrix which might not give expected results! (If you want no translation or rotation, set them to None rather than 0)"
        )
        m = np.linalg.pinv(m)
    return torch.from_numpy(m)


def get_hook(G_synth, layer, size, strategy):
    size = np.flip(size)  # W,H --> H,W

    if strategy == "stretch":

        def hook(module, input, output):
            return interpolate(output, tuple(size), mode="bicubic", align_corners=False)

        return hook

    elif strategy == "pad-zero":
        original_size = getattr(G_synth, G_synth.layer_names[max(layer - 1, 0)]).out_size
        pad_h, pad_w = (size - original_size).astype(int) // 2
        padding = (pad_w, pad_w, pad_h, pad_h)

        def hook(module, input, output):
            return pad(output, padding, mode="constant", value=0)

        return hook

    else:
        raise Exception(f"Resize strategy not found: {strategy}")
