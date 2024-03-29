import torch

from ....GAN.wrappers.stylegan2 import StyleGAN2, StyleGAN2Mapper, StyleGAN2Synthesizer
from . import MauaPatch


class StyleGAN2Patch(MauaPatch):
    """ """

    def __init__(
        self,
        model_file,
        audio_file,
        fps=24,
        offset=0,
        duration=-1,
        output_size=(1024, 1024),
        resize_strategy="pad-zero",
        resize_layer=0,
        inference=False,
    ):
        super().__init__(audio_file, fps, offset, duration)
        self.stylegan2 = StyleGAN2(model_file, inference, output_size, resize_strategy, resize_layer)
        self.mapper = self.stylegan2.mapper
        self.synthesizer = self.stylegan2.synthesizer

    def process_mapper_inputs(self):
        """
        Returns: {
            "latent_z"           : [description]
            "truncation"         : [description]
            "class_conditioning" : [description]
        }
        """
        return {"latent_z": torch.randn((1, 512))}

    def process_synthesizer_inputs(self, latent_w):
        """
        Returns: {
            "latent_w"      : [description]
            "latent_w_plus" : [description]
            "translation"   : [description]
            "rotation"      : [description]
        }
        """
        return latent_w

    def process_outputs(self, video):
        """
        Returns:
            video : [description]
        """
        return video
