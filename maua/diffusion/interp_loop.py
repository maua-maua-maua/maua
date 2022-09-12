import torch
from maua.prompt import ImagePrompt

torch.cuda.is_available()

# torch MUST be imported before decord for reasons?!

import decord

decord.bridge.set_bridge("torch")

import sys

import numpy as np
from tqdm import tqdm

from ..ops.video import VideoWriter
from .image import get_diffusion_model


def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    p = p.permute(2, 0, 1)[..., None]
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a[None] * torch.cos(p) + c[None] * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d


video = decord.VideoReader(sys.argv[1])
fps = video.get_avg_fps()
size, _, _ = video[0].shape

n_frames = 120
timesteps = 25
skip = 0.0  # TODO non-zero doesn't work yet
batch_size = 4

with torch.autocast("cuda"), torch.inference_mode():
    print("loading diffusion model...")
    diffusion = get_diffusion_model(
        diffusion="stable", timesteps=timesteps, sampler="dpm_2", cfg_scale=1, image="yes pls"
    )

    print("calculating prompt interpolation...")
    start = ImagePrompt(img=video[1].permute(2, 0, 1).unsqueeze(0)).img.float().cuda()
    start_cond = diffusion.model.get_learned_conditioning(start)
    latent_shape = tuple(diffusion.model.get_first_stage_encoding(diffusion.model.encode_first_stage(start)).shape)

    end = ImagePrompt(img=video[-2].permute(2, 0, 1).unsqueeze(0)).img.float().cuda()
    end_cond = diffusion.model.get_learned_conditioning(end)

    conds = slerp(end_cond, start_cond, torch.linspace(0, 1, n_frames).to(end_cond))

    # latent input should be reverse sampled interpolation b/w start & end latents rather than fixed noise
    print("diffusing interpolation...")
    latents = torch.cat(
        [
            diffusion.sample_fn(
                diffusion.model_fn,
                torch.from_numpy(np.random.RandomState(42).randn(batch_size, *latent_shape[1:])),
                diffusion.sigmas[round(skip * timesteps) :],
                extra_args={"cond": cond, "uncond": torch.zeros_like(cond), "cond_scale": 1},
                disable=True,
            )
            for cond in conds
        ]
    )

    print("decoding latents...")
    with VideoWriter(output_file="output/interpolated.mp4", output_size=(size, size), fps=fps) as video:
        for latent_batch in tqdm(latents.split(batch_size), unit_scale=batch_size):
            for frame in diffusion.model.decode_first_stage(latent_batch):
                video.write(frame.unsqueeze(0))
