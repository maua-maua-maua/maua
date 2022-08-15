from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

import torch

from ..grad import CLIPGrads, ColorMatchGrads, LPIPSGrads, VGGGrads
from ..ops.image import match_histogram, sharpen
from ..ops.io import save_image
from ..prompt import StylePrompt
from .multires import MultiResolutionDiffusionProcessor
from .processors.glid3xl import GLID3XL
from .processors.glide import GLIDE
from .processors.guided import GuidedDiffusion
from .processors.latent import LatentDiffusion


def build_output_name(init=None, style=None, text=None):
    out_name = str(uuid4())[:6]
    if text is not None:
        out_name = f"{text.replace(' ','_')}_{out_name}"
    if style is not None:
        out_name = f"{Path(style).stem}_{out_name}"
    if init is not None:
        out_name = f"{Path(init).stem}_{out_name}"
    return out_name


def width_height(arg: str):
    w, h = arg.split(",")
    return int(w), int(h)


@torch.no_grad()
def main(
    init: str = "random",
    text: str = None,
    content: Optional[str] = None,
    style: Optional[str] = None,
    sizes: List[Tuple[int, int]] = [(512, 512)],
    skips: List[float] = [0.0],
    timesteps: int = 50,
    super_res: str = "SwinIR-M-DFO-GAN",
    stitch: bool = False,
    tile_size: Optional[int] = None,
    max_batch: int = 4,
    diffusion: str = "guided",
    sampler: str = "plms",
    guidance_speed: str = "fast",
    clip_scale: float = 2500.0,
    lpips_scale: float = 0.0,
    style_scale: float = 0.0,
    color_match_scale: float = 0.0,
    cfg_scale: float = 5.0,
    match_hist: bool = False,
    sharpness: float = 0.0,
    device: str = "cuda",
):
    if diffusion == "guided":
        diffusion = GuidedDiffusion(
            [
                CLIPGrads(scale=clip_scale),
                LPIPSGrads(scale=lpips_scale),
                VGGGrads(scale=style_scale),
                ColorMatchGrads(scale=color_match_scale),
            ],
            sampler=sampler,
            timesteps=timesteps,
            speed=guidance_speed,
        )
    elif diffusion == "latent":
        diffusion = LatentDiffusion(cfg_scale=cfg_scale, sampler=sampler, timesteps=timesteps)
    elif diffusion == "glide":
        diffusion = GLIDE(cfg_scale=cfg_scale, sampler=sampler, timesteps=timesteps)
    elif diffusion == "glid3xl":
        diffusion = GLID3XL(cfg_scale=cfg_scale, sampler=sampler, timesteps=timesteps)
    else:
        raise NotImplementedError()

    pre_hook = partial(match_histogram, source_tensor=StylePrompt(path=style).img) if match_hist else None
    post_hook = partial(sharpen, strength=sharpness) if sharpness > 0 else None

    schedule = {shape: skip for shape, skip in zip(sizes, skips)}

    img = MultiResolutionDiffusionProcessor()(
        diffusion=diffusion.to(device),
        init=init,
        text=text,
        content=content,
        style=style,
        schedule=schedule,
        pre_hook=pre_hook,
        post_hook=post_hook,
        super_res_model=super_res,
        tile_size=tile_size,
        stitch=stitch,
        max_batch=max_batch,
    )

    return img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=True)
    parser.add_argument("--init", type=str, default="random")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--content", type=str, default=None)
    parser.add_argument("--style", type=str, default=None)
    parser.add_argument("--sizes", type=width_height, nargs="+", default=(512, 512))
    parser.add_argument("--skips", type=float, nargs="+", default=0)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--super-res", type=str, default="SwinIR-M-DFO-GAN")
    parser.add_argument("--stitch", action="store_true")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--diffusion", type=str, default="guided", choices=["guided", "latent", "glide", "glid3xl"])
    parser.add_argument("--sampler", type=str, default="plms", choices=["p", "ddim", "plms"])
    parser.add_argument("--guidance-speed", type=str, default="fast", choices=["regular", "fast", "hyper"])
    parser.add_argument("--clip-scale", type=float, default=2500.0)
    parser.add_argument("--lpips-scale", type=float, default=0.0)
    parser.add_argument("--style-scale", type=float, default=0.0)
    parser.add_argument("--color-match-scale", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--match-hist", action="store_true")
    parser.add_argument("--sharpness", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="output/")
    args = parser.parse_args()

    out_name = build_output_name(args.init, args.style, args.text)[:222]
    out_dir = args.out_dir
    del args.out_dir

    img = main(**vars(args))

    save_image(img, f"{out_dir}/{out_name}.png")
