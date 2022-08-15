from typing import Callable, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from resize_right import resize
from resize_right.interp_methods import lanczos3
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from maua.diffusion.processors.base import BaseDiffusionProcessor

from ..ops.image import destitch, restitch
from ..ops.noise import create_perlin_noise
from ..prompt import ContentPrompt, StylePrompt, TextPrompt
from ..super.image.single import upscale_image


def round64(x):
    return round(x / 64) * 64


def get_start_steps(skips, diffusion):
    start_steps = np.argmax(
        diffusion.original_num_steps * (1 - np.array(skips)[:, None]) <= np.array(diffusion.timestep_map)[None, :],
        axis=1,
    )
    return start_steps


def initialize_image(init, shape):
    if init == "random":
        img = torch.randn((1, 3, *shape))
    elif init is not None:
        img = resize(to_tensor(Image.open(init).convert("RGB")).unsqueeze(0).mul(2).sub(1), out_shape=shape)
    elif init == "perlin":
        img = (
            resize(create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False), out_shape=shape)
            + resize(create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True), out_shape=shape)
            - 1
        ).unsqueeze(0)
    else:
        raise Exception("init strategy not recognized!")
    return img


class MultiResolutionDiffusionProcessor(torch.nn.Module):
    def forward(
        self,
        diffusion: BaseDiffusionProcessor,
        init: str,
        text: str = None,
        content: str = None,
        style: str = None,
        schedule: Dict[Tuple[int, int], float] = {(512, 512), 0.5},
        pre_hook: Callable = None,
        post_hook: Callable = None,
        super_res_model: str = None,
        tile_size: int = None,
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
