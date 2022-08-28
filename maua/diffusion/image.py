import traceback
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from maua.diffusion.processors.base import BaseDiffusionProcessor
from PIL import Image
from resize_right import resize
from resize_right.interp_methods import lanczos3
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from ..grad import CLIPGrads, ColorMatchGrads, LPIPSGrads, VGGGrads
from ..ops.image import destitch, match_histogram, restitch, sharpen
from ..ops.io import save_image
from ..ops.noise import create_perlin_noise
from ..prompt import ContentPrompt, StylePrompt, TextPrompt
from ..super.image.single import upscale_image
from .processors.base import BaseDiffusionProcessor
from .processors.glid3xl import GLID3XL
from .processors.glide import GLIDE
from .processors.guided import GuidedDiffusion
from .processors.latent import LatentDiffusion
from .processors.stable import StableDiffusion


def round64(x):
    return round(x / 64) * 64


def width_height(arg: str):
    w, h = arg.split(",")
    return int(w), int(h)


def build_output_name(init=None, style=None, text=None, unique=True):
    out_name = str(uuid4())[:6] if unique else "video"
    if text is not None:
        out_name = f"{text.replace(' ','_')}_{out_name}"
    if style is not None:
        out_name = f"{Path(style).stem}_{out_name}"
    if init is not None:
        out_name = f"{Path(init).stem}_{out_name}"
    return out_name


def get_start_steps(skips, diffusion):
    start_steps = np.argmax(
        diffusion.original_num_steps * (1 - np.array(skips)[:, None])
        <= np.array(list(diffusion.timestep_map[1:]) + [diffusion.original_num_steps])[None, :],
        axis=1,
    )
    return start_steps


def initialize_image(init, shape):
    if init == "random":
        img = torch.randn((1, 3, *shape))
    elif init == "perlin":
        img = (
            resize(create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False), out_shape=shape)
            + resize(create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True), out_shape=shape)
            - 1
        ).unsqueeze(0)
    elif init is not None:
        img = resize(to_tensor(Image.open(init).convert("RGB")).unsqueeze(0).mul(2).sub(1), out_shape=shape)
    else:
        raise Exception("init strategy not recognized!")
    return img


def get_diffusion_model(
    diffusion: Union[str, BaseDiffusionProcessor] = "guided",
    timesteps: int = 50,
    sampler: str = "plms",
    guidance_speed: str = "fast",
    clip_scale: float = 0.0,
    lpips_scale: float = 0.0,
    style_scale: float = 0.0,
    color_match_scale: float = 0.0,
    cfg_scale: float = 5.0,
):
    grad_modules = (
        ([CLIPGrads(scale=clip_scale)] if clip_scale > 0 else [])
        + ([LPIPSGrads(scale=lpips_scale)] if lpips_scale > 0 else [])
        + ([VGGGrads(scale=style_scale)] if style_scale > 0 else [])
        + ([ColorMatchGrads(scale=color_match_scale)] if color_match_scale > 0 else [])
    )

    if diffusion == "guided":
        diffusion = GuidedDiffusion(
            grad_modules=grad_modules, sampler=sampler, timesteps=timesteps, speed=guidance_speed
        )
    elif diffusion == "latent":
        diffusion = LatentDiffusion(cfg_scale=cfg_scale, sampler=sampler, timesteps=timesteps)
    elif diffusion == "stable":
        diffusion = StableDiffusion(
            grad_modules=grad_modules, cfg_scale=cfg_scale, sampler=sampler, timesteps=timesteps
        )
    elif diffusion == "glide":
        diffusion = GLIDE(cfg_scale=cfg_scale, sampler=sampler, timesteps=timesteps)
    elif diffusion == "glid3xl":
        diffusion = GLID3XL(cfg_scale=cfg_scale, sampler=sampler, timesteps=timesteps)
    else:
        try:
            diffusion = StableDiffusion(
                grad_modules=grad_modules,
                model_checkpoint=diffusion,
                cfg_scale=cfg_scale,
                sampler=sampler,
                timesteps=timesteps,
            )
        except:
            traceback.print_exc()
        assert isinstance(diffusion, BaseDiffusionProcessor)
    return diffusion


class MultiResolutionDiffusionProcessor(torch.nn.Module):
    def forward(
        self,
        diffusion: BaseDiffusionProcessor,
        init: str,
        text: Optional[str] = None,
        content: Optional[str] = None,
        style: Optional[str] = None,
        schedule: Dict[Tuple[int, int], float] = {(512, 512), 0.5},
        pre_hook: Optional[Callable] = None,
        post_hook: Optional[Callable] = None,
        super_res_model: Optional[str] = None,
        tile_size: Optional[int] = None,
        stitch: bool = True,
        max_batch: int = 4,
        verbose: bool = True,
    ):
        shapes = [(round64(h), round64(w)) for h, w in list(schedule.keys())]
        start_steps = get_start_steps(list(schedule.values()), diffusion)

        if tile_size is None:
            tile_size = diffusion.image_size

        # initialize image
        img = initialize_image(init, shapes[0])
        if content is None:
            content = dict(img=img.clone())
        else:
            content = dict(path=content)

        for scale, start_step in enumerate(start_steps):
            if verbose:
                print(f"Current size: {shapes[scale][1]}x{shapes[scale][0]}")

            if scale != 0:
                # maybe upsample image with super-resolution model
                if super_res_model:
                    img = upscale_image(img.add(1).div(2), model_name=super_res_model).mul(2).sub(1)

                # resize image for next scale
                img = resize(img, out_shape=shapes[scale], interp_method=lanczos3).cpu()

            if pre_hook:  # user-supplied pre-processing function
                img = pre_hook(img)

            # if the image is larger than specified size, chop it into tiles
            needs_stitching = stitch and min(shapes[scale]) > tile_size
            if needs_stitching:
                img = destitch(img, tile_size=tile_size)

            # initialize prompts for diffusion (we don't support stitched content yet)
            prompts = [ContentPrompt(**content).to(img)] if not needs_stitching else []
            if text is not None:
                prompts.append(TextPrompt(text))
            if style is not None:
                prompts.append(StylePrompt(path=style, size=shapes[scale]))

            # run diffusion sampling (in multiple batches if necessary)
            dev = diffusion.device
            if img.shape[0] > max_batch:
                tiles = tqdm(img.split(max_batch)) if verbose else img.split(max_batch)
                img = torch.cat([diffusion(ims.to(dev), prompts, start_step, verbose=False) for ims in tiles])
            else:
                img = diffusion(img.to(dev), prompts, start_step, verbose=verbose)

            # reassemble image tiles to final image
            if needs_stitching:
                img = restitch(img, *shapes[scale])

            if post_hook:  # user-supplied post-processing function
                img = post_hook(img)

        return img


@torch.no_grad()
def image_sample(
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
    diffusion: Union[str, BaseDiffusionProcessor] = "guided",
    sampler: str = "plms",
    guidance_speed: str = "fast",
    clip_scale: float = 0.0,
    lpips_scale: float = 0.0,
    style_scale: float = 0.0,
    color_match_scale: float = 0.0,
    cfg_scale: float = 5.0,
    match_hist: bool = False,
    sharpness: float = 0.0,
    device: str = "cuda",
    number: int = 1,
):
    diffusion = get_diffusion_model(
        diffusion=diffusion,
        timesteps=timesteps,
        sampler=sampler,
        guidance_speed=guidance_speed,
        clip_scale=clip_scale,
        lpips_scale=lpips_scale,
        style_scale=style_scale,
        color_match_scale=color_match_scale,
        cfg_scale=cfg_scale,
    )

    pre_hook = partial(match_histogram, source_tensor=StylePrompt(path=style).img) if match_hist else None
    post_hook = partial(sharpen, strength=sharpness) if sharpness > 0 else None

    assert len(sizes) == len(skips), "`sizes` and `skips` must have equal length!"
    schedule = {shape: skip for shape, skip in zip(sizes, skips)}

    if number > 1:
        for _ in range(number):
            yield MultiResolutionDiffusionProcessor()(
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
    else:
        return MultiResolutionDiffusionProcessor()(
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


if __name__ == "__main__":
    # fmt:off
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=True)
    parser.add_argument("--init", type=str, default="random", help='How to initialize the image "random", "perlin", or a path to an image file.')
    parser.add_argument("--text", type=str, default=None, help='A text prompt to visualize.')
    parser.add_argument("--content", type=str, default=None, help='A content image whose structure to adapt in the output image (only works with "guided" diffusion at the moment, see --lpips-scale).')
    parser.add_argument("--style", type=str, default=None, help='An image whose style should be optimized for in the output image (only works with "guided" diffusion at the moment, see --style-scale).')
    parser.add_argument("--sizes", type=width_height, nargs="+", default=[(512, 512)], help='Sequence of sizes to synthesize the image at.')
    parser.add_argument("--skips", type=float, nargs="+", default=[0], help='Sequence of skip fractions for each size. Lower fractions will stray further from the original image, while higher fractions will hallucinate less detail.')
    parser.add_argument("--timesteps", type=int, default=50, help='Number of timesteps to sample the diffusion process at. Higher values will take longer but are generally of higher quality.')
    parser.add_argument("--super-res", type=str, default="SwinIR-M-DFO-GAN", help='Super resolution model to upscale intermediate results with before applying next diffusion resolution (see maua.super.image --model-help for full list of possibilities, None to perform simple resizing).')
    parser.add_argument("--stitch", action="store_true", help='Enable tiled synthesis of images which are larger than the specified --tile-size.')
    parser.add_argument("--tile-size", type=int, default=None, help='The maximum size of tiles the image is cut into.')
    parser.add_argument("--max-batch", type=int, default=4, help='Maximum batch of tiles to synthesize at one time (lower values use less memory, but will be slower).')
    parser.add_argument("--diffusion", type=str, default="stable", help='Which diffusion model to use. Options: "guided", "latent", "glide", "glid3xl", "stable" or a /path/to/stable-diffusion.ckpt')
    parser.add_argument("--sampler", type=str, default="dpm_2", choices=["p", "ddim", "plms", "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms"], help='Which sampling method to use. "p", "ddim", and "plms" work for all diffusion models, the rest are currently only supported with "stable" diffusion.')
    parser.add_argument("--guidance-speed", type=str, default="fast", choices=["regular", "fast"], help='How to perform "guided" diffusion. "regular" is slower but can be higher quality, "fast" corresponds to the secondary model method (a.k.a. Disco Diffusion).')
    parser.add_argument("--clip-scale", type=float, default=0.0, help='Controls strength of CLIP guidance when using "guided" diffusion.')
    parser.add_argument("--lpips-scale", type=float, default=0.0, help='Controls the apparent influence of the content image when using "guided" diffusion and a --content image.')
    parser.add_argument("--style-scale", type=float, default=0.0, help='When using "guided" diffusion and a --style image, a higher --style-scale enforces textural similarity to the style, while a lower value will be conceptually similar to the style.')
    parser.add_argument("--color-match-scale", type=float, default=0.0, help='When using "guided" diffusion, the --color-match-scale guides the output\'s colors to match the --style image.')
    parser.add_argument("--cfg-scale", type=float, default=7.5, help='Classifier-free guidance strength. Higher values will match the text prompt more closely at the cost of output variability.')
    parser.add_argument("--match-hist", action="store_true", help='Match the histogram of the initialization image to the --style image before starting diffusion.')
    parser.add_argument("--sharpness", type=float, default=0.0, help='Sharpen the image by this amount after each diffusion scale (a value of 1.0 will leave the image unchanged, higher values will be sharper).')
    parser.add_argument("--device", type=str, default="cuda", help='Which device to use (e.g. "cpu" or "cuda:1")')
    parser.add_argument("--number", type=int, default=1, help='How many images to render.')
    parser.add_argument("--out-dir", type=str, default="output/", help='Directory to save output images to.')
    args = parser.parse_args()
    # fmt:on

    out_name = build_output_name(args.init, args.style, args.text)[:222]
    out_dir = args.out_dir
    del args.out_dir

    for i, img in enumerate(image_sample(**vars(args))):
        save_image(img, f"{out_dir}/{out_name}{i}.png")
