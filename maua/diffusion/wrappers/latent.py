import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from ...diffusion.conditioning import TextPrompt
from ...utility import download
from .base import DiffusionWrapper

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../../submodules/latent_diffusion")
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
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
                txt, scale = prompt()
                conditioning = self.model.get_learned_conditioning([txt])
                unconditional = self.model.get_learned_conditioning([""]) if scale != 1 else None
        return conditioning, unconditional, scale


class LatentDiffusion(DiffusionWrapper):
    def __init__(
        self,
        sampler="ddim",
        timesteps=100,
        model_checkpoint="large",
        ddim_eta=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()

        self.model = get_model(model_checkpoint)

        self.conditioning = LatentConditioning(self.model)

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
        self.timestep_map = np.linspace(0, sampler.ddpm_num_timesteps, timesteps + 1).round().astype(np.long)

    @torch.no_grad()
    def sample(self, img, prompts, start_step, n_steps=None, verbose=True):
        if n_steps is None:
            n_steps = start_step

        conditioning, unconditional, scale = self.conditioning([p.to(img) for p in prompts])

        with self.model.ema_scope():
            x_T = self.model.get_first_stage_encoding(self.model.encode_first_stage(img))

            t = torch.ones([x_T.shape[0]], device=self.device, dtype=torch.long) * self.timestep_map[start_step]
            x_T = self.model.q_sample(x_T, t - 1, torch.randn_like(x_T))

            samples, _ = self.sample_fn(
                x_T=x_T,
                shape=x_T.shape,
                timesteps=n_steps,
                cond=conditioning.tile(x_T.shape[0], 1, 1),
                unconditional_guidance_scale=scale,
                unconditional_conditioning=unconditional.tile(x_T.shape[0], 1, 1),
            )
            samples_out = self.model.decode_first_stage(samples)

        return samples_out
