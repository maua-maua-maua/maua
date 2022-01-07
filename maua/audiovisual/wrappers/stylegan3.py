import sys
import warnings
from typing import Tuple

import numpy as np
import torch
from torch.nn.functional import interpolate, pad

from . import MauaMapper, MauaSynthesizer

sys.path.append("maua/GAN")
from maua.GAN import dnnlib, legacy
from maua.GAN.training.networks_stylegan3 import (MappingNetwork,
                                                  SynthesisNetwork)

layer_multipliers = {
    1024: {0: 64, 1: 64, 2: 64, 3: 32, 4: 32, 5: 16, 6: 8, 7: 8, 8: 4, 9: 4, 10: 2, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1},
    512: {0: 32, 1: 32, 2: 32, 3: 16, 4: 16, 5: 8, 6: 8, 7: 4, 8: 4, 9: 2, 10: 2, 11: 1, 12: 1, 13: 1, 14: 1},
    256: {0: 16, 1: 16, 2: 16, 3: 16, 4: 8, 5: 8, 6: 4, 7: 4, 8: 2, 9: 2, 10: 2, 11: 1, 12: 1, 13: 1, 14: 1},
}


class StyleGAN3:
    def __init__(self, mapper, synthesizer) -> None:
        self.mapper = mapper
        self.synthesizer = synthesizer

    def get_z_latents(self, seeds):
        seeds = sum(
            [
                ([int(seed)] if not "-" in seed else list(range(int(seed.split("-")[0]), int(seed.split("-")[1]))))
                for seed in seeds.split(",")
            ],
            [],
        )
        latent_z = torch.cat(
            [torch.from_numpy(np.random.RandomState(seed).randn(1, self.mapper.z_dim)) for seed in seeds]
        )
        return latent_z

    def get_w_latents(self, seeds, truncation=1):
        latent_z = self.get_z_latents(seeds)
        latent_w = self.mapper(latent_z, truncation=truncation)
        return latent_w


class StyleGAN3Mapper(MauaMapper):
    def __init__(self, model_file: str) -> None:
        super().__init__()

        with dnnlib.util.open_url(model_file) as f:
            self.G_map: MappingNetwork = legacy.load_network_pkl(f)["G_ema"].mapping
        self.z_dim, self.c_dim = self.G_map.z_dim, self.G_map.c_dim

        self.modulation_targets = {
            "latent_z": (self.z_dim,),
            "truncation": (1,),
        }
        if self.c_dim > 0:
            self.modulation_targets["class_conditioning"] = (self.c_dim,)

    def forward(
        self,
        latent_z: torch.Tensor,
        class_conditioning: torch.Tensor = None,
        truncation: torch.Tensor = torch.ones(1, 1),
    ):
        return self.G_map.forward(latent_z, class_conditioning, truncation_psi=truncation)


class StyleGAN3Synthesizer(MauaSynthesizer):
    def __init__(self, model_file: str, output_size: Tuple[int, int], strategy: str, layer: int) -> None:
        super().__init__()

        with dnnlib.util.open_url(model_file) as f:
            self.G_synth: SynthesisNetwork = legacy.load_network_pkl(f)["G_ema"].synthesis
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
        translation: torch.Tensor = torch.zeros(1, 2),
        rotation: torch.Tensor = torch.zeros(1, 1),
    ) -> torch.Tensor:
        if latent_w is None and latent_w_plus is None:
            raise Exception("One of latent_w or latent_w_plus inputs must be supplied!")
        if latent_w is not None and latent_w_plus is not None:
            warnings.warn("Both latent_w and latent_w_plus supplied, using latent_w_plus input...")

        if not (torch.all(translation == 0) and rotation == 0):
            self.G_synth.input.transform.copy_(make_transform_mat(translation, rotation))

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


def make_transform_mat(translate: Tuple[float, float], angle: float) -> torch.Tensor:
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m = np.array([[c, s, translate[0]], [-s, c, translate[1]], [0, 0, 0]])
    m = np.linalg.inv(m)
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
