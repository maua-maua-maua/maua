import os
import sys
from functools import partial

import numpy as np
import torch
from resize_right import resize
from tqdm import trange

from ..conditioning import TextPrompt
from .base import DiffusionWrapper

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../../submodules/GLIDE/")

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)


class GLIDE(DiffusionWrapper):
    def __init__(
        self,
        sampler="plms",
        timesteps=25,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ddim_eta=0,
        temp=1.0,
    ):
        super().__init__()
        self.temp, self.device = temp, device

        using_gpu = device.type == "cuda"

        # Create base model.
        options = model_and_diffusion_defaults()
        options["use_fp16"] = using_gpu
        options["timestep_respacing"] = str(timesteps + 2)
        model, diffusion = create_model_and_diffusion(**options)
        model.eval()
        if using_gpu:
            model.convert_to_fp16()
        model.to(device)
        model.load_state_dict(load_checkpoint("base", device, cache_dir="modelzoo/"))
        self.model, self.diffusion = model, diffusion
        self.ctx, self.original_num_steps = options["text_ctx"], options["diffusion_steps"]
        self.timestep_map = np.linspace(0, self.original_num_steps, timesteps + 1).round().astype(np.long)

        # Create upsampler model.
        options_up = model_and_diffusion_defaults_upsampler()
        options_up["use_fp16"] = using_gpu
        options_up["timestep_respacing"] = str(round(0.6 * timesteps) + 2)
        model_up, diffusion_up = create_model_and_diffusion(**options_up)
        model_up.eval()
        if using_gpu:
            model_up.convert_to_fp16()
        model_up.to(device)
        model_up.load_state_dict(load_checkpoint("upsample", device, cache_dir="modelzoo/"))
        self.model_up, self.diffusion_up = model_up, diffusion_up
        self.scale_factor = options_up["image_size"] / options["image_size"]

        def model_fn(x_t, ts, scale, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        if sampler == "p":
            self.sample_fn = lambda _, scale: partial(self.diffusion.p_sample, model=partial(model_fn, scale=scale))
            self.upsample_fn = self.diffusion_up.p_sample_loop
        elif sampler == "ddim":
            self.sample_fn = lambda _, scale: partial(
                self.diffusion.ddim_sample, eta=ddim_eta, model=partial(model_fn, scale=scale)
            )
            self.upsample_fn = self.diffusion_up.ddim_sample_loop
        elif sampler == "plms":
            self.sample_fn = lambda old_eps, scale: partial(
                (
                    self.diffusion.prk_sample
                    if len(old_eps) < 3
                    else partial(self.diffusion.plms_sample, old_eps=old_eps)
                ),
                model=partial(model_fn, scale=scale),
            )
            self.upsample_fn = self.diffusion_up.plms_sample_loop
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def sample(self, img, prompts, start_step, n_steps=None, verbose=True):
        B, C, H, W = img.shape

        for prompt in prompts:
            if isinstance(prompt, TextPrompt):
                txt, scale = prompt()
                tokens = self.model.tokenizer.encode(txt)
                tokens, mask = self.model.tokenizer.padded_tokens_and_mask(tokens, self.ctx)
                uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask([], self.ctx)
                model_kwargs = dict(
                    tokens=torch.tensor([tokens] * B + [uncond_tokens] * B, device=self.device),
                    mask=torch.tensor([mask] * B + [uncond_mask] * B, dtype=torch.bool, device=self.device),
                )
                break

        if n_steps is None:
            n_steps = start_step

        img = resize(img, scale_factors=1 / self.scale_factor)
        noise = torch.randn_like(img)
        img = self.diffusion.q_sample(img, torch.tensor([start_step] * B, device=self.device, dtype=torch.long), noise)
        img = torch.cat((img, noise))

        t = torch.tensor([start_step] * img.shape[0], device=self.device, dtype=torch.long)

        self.model.del_cache()
        old_eps = []
        for _ in (trange if verbose else range)(n_steps):
            out = self.sample_fn(old_eps, scale)(x=img, t=t, model_kwargs=model_kwargs)
            img = out["sample"]

            if "eps" in out:  # PLMS bookkeeping
                if len(old_eps) >= 3:
                    old_eps.pop(0)
                old_eps.append(out["eps"])

            t -= 1
        self.model.del_cache()

        if t.sum() == 0:
            tokens = self.model_up.tokenizer.encode(txt)
            tokens, mask = self.model_up.tokenizer.padded_tokens_and_mask(tokens, self.ctx)
            model_kwargs = dict(
                low_res=((out["sample"][:B] + 1) * 127.5).round() / 127.5 - 1,
                tokens=torch.tensor([tokens] * B, device=self.device),
                mask=torch.tensor([mask] * B, dtype=torch.bool, device=self.device),
            )
            self.model_up.del_cache()
            up_shape = (B, C, H, W)
            final = self.upsample_fn(
                self.model_up,
                up_shape,
                noise=torch.randn(up_shape, device=self.device) * self.temp,
                device=self.device,
                model_kwargs=model_kwargs,
                progress=verbose,
            )[:B]
            self.model_up.del_cache()
        else:
            final = resize(out["pred_xstart"][:B], scale_factors=self.scale_factor)

        return final
