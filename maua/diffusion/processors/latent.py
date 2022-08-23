import os
import sys
import warnings

import numpy as np
import torch
from omegaconf import OmegaConf

from ...prompt import TextPrompt
from ...utility import download
from .base import BaseDiffusionProcessor

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/../../submodules/VQGAN")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/../../submodules/latent_diffusion")
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config


class Silence:
    def __enter__(self):
        from transformers import logging

        logging.set_verbosity_error()
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        from transformers import logging

        logging.set_verbosity_warning()


def load_model_from_config(config, ckpt):
    with Silence():
        sd = torch.load(ckpt, map_location="cpu")["state_dict"]
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        model.cuda()
        model.eval()
    return model


def get_model(checkpoint):
    if checkpoint == "large":
        ckpt = "modelzoo/latent-diffusion-text2img-large.ckpt"
        config = "maua/submodules/latent_diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        if not os.path.exists(ckpt):
            download("https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt", ckpt)
    else:
        ckpt = checkpoint
        config = checkpoint.replace(".ckpt", ".yaml")
    return load_model_from_config(OmegaConf.load(config), ckpt)


class LatentConditioning(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, prompts):
        for prompt in prompts:
            if isinstance(prompt, TextPrompt):
                txt, _ = prompt()
                conditioning = self.model.get_learned_conditioning([txt])
                unconditional = self.model.get_learned_conditioning([""])
        return conditioning, unconditional


class LatentDiffusion(BaseDiffusionProcessor):
    def __init__(
        self,
        cfg_scale=3,
        sampler="ddim",
        timesteps=100,
        model_checkpoint="large",
        ddim_eta=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()

        self.model = get_model(model_checkpoint)

        self.conditioning = LatentConditioning(self.model)
        self.cfg_scale = cfg_scale

        if sampler == "plms":
            sampler = PLMSSampler(self.model)
            sampler.make_schedule(ddim_num_steps=timesteps, ddim_eta=ddim_eta, verbose=False)
            self.sample_fn = sampler.plms_sampling
        else:
            sampler = DDIMSampler(self.model)
            sampler.make_schedule(ddim_num_steps=timesteps, ddim_eta=ddim_eta, verbose=False)
            self.sample_fn = sampler.ddim_sampling

        self.device = device
        self.model = self.model.to(device)
        self.original_num_steps = sampler.ddpm_num_timesteps
        self.timestep_map = np.linspace(0, sampler.ddpm_num_timesteps, timesteps + 1).round().astype(int)
        self.image_size = self.model.image_size * 8

    @torch.no_grad()
    def forward(self, img, prompts, start_step, n_steps=None, verbose=True):
        if n_steps is None:
            n_steps = start_step

        conditioning, unconditional = self.conditioning([p.to(img) for p in prompts])

        with self.model.ema_scope():
            x_T = self.model.get_first_stage_encoding(self.model.encode_first_stage(img))

            t = torch.ones([x_T.shape[0]], device=self.device, dtype=torch.long) * self.timestep_map[start_step]
            x_T = self.model.q_sample(x_T, t - 1, torch.randn_like(x_T))

            samples, _ = self.sample_fn(
                x_T=x_T,
                shape=x_T.shape,
                timesteps=n_steps,
                cond=conditioning.tile(x_T.shape[0], 1, 1),
                unconditional_guidance_scale=self.cfg_scale,
                unconditional_conditioning=unconditional.tile(x_T.shape[0], 1, 1) if self.cfg_scale != 1 else None,
            )
            samples_out = self.model.decode_first_stage(samples)

        return samples_out
