from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..load import load_network
from . import MauaMapper


class StyleGAN(torch.nn.Module):
    def __init__(self, mapper, synthesizer) -> None:
        super().__init__()
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

    def forward(self, z):
        return self.synthesizer(self.mapper(z))


class StyleGANMapper(MauaMapper):
    MappingNetwork = None

    def __init__(self, model_file: str, inference: bool) -> None:
        super().__init__()

        if model_file is None or model_file == "None":
            self.G_map = self.__class__.MappingNetwork(inference)(z_dim=512, c_dim=0, w_dim=512, num_ws=18)
        else:
            self.G_map = load_network(model_file, inference).mapping

        self.z_dim, self.c_dim = self.G_map.z_dim, self.G_map.c_dim

        self.modulation_targets = {
            "latent_z": (self.z_dim,),
            "truncation": (1,),
        }
        if self.c_dim > 0:
            self.modulation_targets["class_conditioning"] = (self.c_dim,)

    def forward(self, latent_z: Tensor, class_conditioning: Optional[Tensor] = None, truncation: float = 1.0):
        return self.G_map.forward(latent_z, class_conditioning, truncation_psi=truncation)
