from typing import Tuple

import torch


class MauaMapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplemented


class MauaSynthesizer(torch.nn.Module):
    _hook_handles = []

    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplemented

    def change_output_resolution(self):
        raise NotImplemented

    def refresh_model_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []


def get_generator_classes(architecture: str) -> Tuple[MauaMapper, MauaSynthesizer]:
    if architecture == "stylegan3":
        from .stylegan3 import StyleGAN3Mapper, StyleGAN3Synthesizer

        return StyleGAN3Mapper, StyleGAN3Synthesizer
    if architecture == "stylegan2":
        from .stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer

        return StyleGAN2Mapper, StyleGAN2Synthesizer
    else:
        raise Exception(f"Architecture not found: {architecture}")
