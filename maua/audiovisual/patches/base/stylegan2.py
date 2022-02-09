from maua.GAN.wrappers.stylegan2 import StyleGAN2, StyleGAN2Mapper, StyleGAN2Synthesizer
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
        self.mapper = StyleGAN2Mapper(model_file, inference)
        self.synthesizer = StyleGAN2Synthesizer(model_file, inference, output_size, resize_strategy, resize_layer)
        self.stylegan2 = StyleGAN2(self.mapper, self.synthesizer)

    def process_mapper_inputs(self):
        """
        Returns: {
            "latent_z"           : [description]
            "truncation"         : [description]
            "class_conditioning" : [description]
        }
        """
        return {}

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
