import numpy as np
import torch


class StyleGAN:
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
