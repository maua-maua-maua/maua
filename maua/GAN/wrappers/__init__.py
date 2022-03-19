import torch


class MauaMapper(torch.nn.Module):
    def forward(self):
        raise NotImplementedError()


class MauaSynthesizer(torch.nn.Module):
    _hook_handles = []

    def forward(self):
        raise NotImplementedError()

    def change_output_resolution(self):
        raise NotImplementedError()

    def refresh_model_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []


class MauaGenerator(torch.nn.Module):
    MapperCls = None
    SynthesizerCls = None

    def __init__(self, mapper_kwargs={}, synthesizer_kwargs={}) -> None:
        super().__init__()
        self.mapper = self.__class__.MapperCls(**mapper_kwargs)
        self.synthesizer = self.__class__.SynthesizerCls(**synthesizer_kwargs)

    def forward(self):
        raise NotImplementedError()


def get_generator_class(architecture: str) -> MauaGenerator:
    if architecture == "stylegan3":
        from .stylegan3 import StyleGAN3

        return StyleGAN3
    if architecture == "stylegan2":
        from .stylegan2 import StyleGAN2

        return StyleGAN2
    else:
        raise Exception(f"Architecture not found: {architecture}")
