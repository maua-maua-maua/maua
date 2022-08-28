import sys
from glob import glob

import torch
from einops import rearrange
from PIL import Image
from torch.nn.functional import interpolate
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from torchvision.transforms.functional import resize, to_tensor
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


images = list(reversed(sorted(glob(sys.argv[1] + "/*"))))
prompt = sys.argv[2]
n_frames = 120
fps = 12
size = 512
timesteps = 100
encode_skip = 0.90
decode_skip = 0.99
batch_size = 4
cfg_scale = 7.5
interp = "slerp"

with torch.autocast("cuda"), torch.inference_mode():

    print("loading diffusion model...")
    diffusion = get_diffusion_model(diffusion="stable", timesteps=timesteps, sampler="dpm_2", cfg_scale=cfg_scale)
    diffusion.model.half()

    cond = diffusion.model.get_learned_conditioning([prompt]).tile(batch_size, 1, 1)
    uncond = diffusion.model.get_learned_conditioning([""]).tile(batch_size, 1, 1)

    def style(latent_batch, skip):
        B = latent_batch.shape[0]
        latent_batch += torch.randn_like(latent_batch) * diffusion.sigmas[round(skip * timesteps)]
        return diffusion.sample_fn(
            diffusion.model_fn,
            latent_batch,
            diffusion.sigmas[round(skip * timesteps) :],
            extra_args={"cond": cond[:B], "uncond": uncond[:B], "cond_scale": cfg_scale},
            disable=True,
        )

    print("encoding images...")
    latents = torch.cat(
        [
            style(
                diffusion.model.get_first_stage_encoding(
                    diffusion.model.encode_first_stage(
                        to_tensor(resize(Image.open(img), size, antialias=True)).cuda().unsqueeze(0)
                    )
                ),
                skip=encode_skip,
            )
            for img in tqdm(images)
        ]
    )

    print("interpolating latents...")
    if interp == "spline":
        t_in = torch.linspace(0, 1, len(latents)).to(latents)
        t_out = torch.linspace(0, 1, n_frames).to(latents)
        latents = rearrange(latents, "t c h w -> (c h) t w")
        interpolated = NaturalCubicSpline(natural_cubic_spline_coeffs(t_in, latents)).evaluate(t_out)
        interpolated = rearrange(interpolated, "(c h) t w -> t c h w", c=4)
    elif interp == "slerp":
        t = torch.linspace(0, 1, round(n_frames / len(latents))).to(latents)
        _, c, h, w = latents.shape
        latents = latents.flatten(1)
        out = slerp(latents[:-1], latents[1:], t)
        out = out.reshape(-1, c * h, w)
        out = interpolate(out.permute(1, 2, 0), size=n_frames, mode="linear", align_corners=False)
        out = out.permute(2, 0, 1).reshape(n_frames, c, h, w)

    print("decoding latents...")
    with VideoWriter(output_file="output/interpolated.mp4", output_size=(size, size), fps=fps) as video:
        for latent_batch in tqdm(interpolated.split(batch_size), unit_scale=batch_size):
            for frame in diffusion.model.decode_first_stage(style(latent_batch, decode_skip)):
                video.write(frame.unsqueeze(0))
