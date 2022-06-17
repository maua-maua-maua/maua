from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..load import load_network
from . import MauaGenerator, MauaMapper, MauaSynthesizer


class StyleGANMapper(MauaMapper):
    MapperClsFn = lambda: None

    def __init__(self, model_file: str, inference: bool) -> None:
        super().__init__()

        if model_file is None or model_file == "None":
            self.G_map = self.__class__.MapperClsFn(inference)(z_dim=512, c_dim=0, w_dim=512, num_ws=18)
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


class StyleGANSynthesizer(MauaSynthesizer):
    pass


class StyleGAN(MauaGenerator):
    __constants__ = ["z_dim", "c_dim", "w_dim", "num_ws", "res", "model_file"]
    MapperCls = StyleGANMapper
    SynthesizerCls = StyleGANSynthesizer

    def __init__(self, model_file=None, inference=False, output_size=(1024, 1024), strategy="stretch", layer=0) -> None:
        super().__init__(
            mapper_kwargs=dict(model_file=model_file, inference=inference),
            synthesizer_kwargs=dict(
                model_file=model_file, inference=inference, output_size=output_size, strategy=strategy, layer=layer
            ),
        )
        self.z_dim = self.mapper.G_map.z_dim
        self.c_dim = self.mapper.G_map.c_dim
        self.w_dim = self.mapper.G_map.w_dim
        self.num_ws = self.mapper.G_map.num_ws
        self.res = self.synthesizer.G_synth.img_resolution
        self.model_file = model_file

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

    def forward(self, z, *args, c=None, **kwargs):
        return self.synthesizer(self.mapper(z, c))
