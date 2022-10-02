import sys

import numpy as np
import torch
from maua.diffusion.processors.stable import StableDiffusion
from maua.prompt import ImagePrompt
from PIL import Image
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


n_frames = 48
timesteps = 100
t_start = 0.5
batch_size = 4
cfg_scale = 1
size = (512, 512)
fps = 24

with torch.autocast("cuda"), torch.inference_mode():
    print("loading diffusion model...")
    diffusion: StableDiffusion = get_diffusion_model(
        diffusion="stable", timesteps=timesteps, sampler="dpm_adaptive", cfg_scale=cfg_scale, image="yes pls"
    )

    print("reverse sampling images...")
    start_prompt = ImagePrompt(path=sys.argv[1], size=size)
    start_img = start_prompt.img.float().cuda()

    for t_start in np.linspace(0, 0.9, 10):

        img_lat = diffusion.encode(start_img)
        print(img_lat.norm().item())
        img_lat_forward = diffusion.forward(img_lat, [start_prompt], t_start=t_start, latent=True)
        print(img_lat_forward.norm().item())
        img_forward = diffusion.decode(img_lat_forward)
        Image.fromarray(
            img_forward.squeeze().permute(1, 2, 0).add(1).div(2).clamp(0, 1).mul(255).byte().cpu().numpy()
        ).save(f"output/forward{t_start:.1f}.png")

        img_lat = diffusion.encode(start_img)
        img_lat_sigma = img_lat + torch.randn_like(img_lat) * diffusion.get_sigmas(t_start)
        print(img_lat_sigma.norm().item())
        img_lat_noise = diffusion.forward(img_lat, [start_prompt], t_start=t_start, reverse=True, latent=True)
        print(img_lat_noise.norm().item())
        img_lat_reverse = diffusion.forward(img_lat_noise, [start_prompt], t_start=t_start, latent=True)
        print(img_lat_reverse.norm().item())
        img_reverse = diffusion.decode(img_lat_reverse)
        Image.fromarray(
            img_reverse.squeeze().permute(1, 2, 0).add(1).div(2).clamp(0, 1).mul(255).byte().cpu().numpy()
        ).save(f"output/reverse{t_start:.1f}.png")
        print()

    exit(0)

    end_prompt = ImagePrompt(path=sys.argv[2], size=size)
    end_img = end_prompt.img.float().cuda()
    end_lat = diffusion.encode(end_img)
    end_noise = diffusion.forward(end_lat, [end_prompt], t_start=t_start, reverse=True, latent=True)

    print("interpolating...")
    interp_coefs = torch.linspace(0, 1, n_frames).to(end_noise)
    inits = slerp(start_noise.flatten(2), end_noise.flatten(2), interp_coefs).reshape(n_frames, 1, *end_noise.shape[1:])

    start_cond = diffusion.model.get_learned_conditioning(start_img)
    end_cond = diffusion.model.get_learned_conditioning(end_img)
    uncond = diffusion.model.get_learned_conditioning(torch.rand_like(end_img).mul(2).sub(1))
    conds = slerp(end_cond, start_cond, interp_coefs)

    # latent input should be reverse sampled interpolation b/w start & end latents rather than fixed noise
    print("diffusing interpolation...")
    latents = torch.cat(
        [
            diffusion.sample_fn(
                diffusion.model_fn,
                init,
                diffusion.get_sigmas(t_start, 1),
                extra_args={"cond": cond, "uncond": uncond, "cond_scale": cfg_scale},
                disable=True,
            )
            for init, cond in zip(tqdm(inits), conds)
        ]
    )

    print("decoding latents...")
    with VideoWriter(output_file="output/interpolated.mp4", output_size=size, fps=fps) as video:
        for latent_batch in tqdm(latents.split(batch_size), unit_scale=batch_size):
            for frame in diffusion.decode(latent_batch):
                video.write(frame.unsqueeze(0))
