# fmt:off
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, ImageOps
from resize_right import resize
from resize_right.interp_methods import lanczos3
from scipy.special import comb
from torchvision.transforms.functional import (adjust_sharpness, autocontrast,
                                               to_pil_image, to_tensor)
from tqdm import tqdm, trange

from ..ops.image import match_histogram
from ..ops.loss import range_loss, tv_loss
from ..super.image.single import upscale_image
from .conditioning import (CLIPGrads, ColorMatchGrads, ContentPrompt,
                           LossGrads, LPIPSGrads, StylePrompt, TextPrompt,
                           VGGGrads)
from .wrappers.glid3xl import GLID3XL
from .wrappers.glide import GLIDE
from .wrappers.guided import GuidedDiffusion
from .wrappers.latent import LatentDiffusion
# fmt:on


def interp(t):
    return 3 * t**2 - 2 * t**3


def perlin(width, height, scale=10, device="cuda"):
    # https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = to_pil_image(out.clamp(0, 1)).convert("RGB")
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = to_pil_image(out.clamp(0, 1).squeeze())
    out = ImageOps.autocontrast(out)
    return to_tensor(out)


def round64(x):
    return round(x / 64) * 64


def save(tensor, filename):
    to_pil_image(tensor.squeeze().add(1).div(2).clamp(0, 1)).save(filename)


def destitch(img, tile_size, overtile=1):
    _, _, H, W = img.shape
    n_rows = round(np.floor(H / tile_size) + overtile)
    n_cols = round(np.floor(W / tile_size) + overtile)
    tiled = []
    for y in torch.linspace(0, H - tile_size, n_rows).round().long():
        for x in torch.linspace(0, W - tile_size, n_cols).round().long():
            tiled.append(img[..., y : y + tile_size, x : x + tile_size])
    return torch.cat(tiled, dim=0)


def smoothstep(x, N=2):
    result = torch.zeros_like(x)
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
    result *= x ** (N + 1)
    return result


def blend_weight1d(total_size, fade_in, fade_out):
    return torch.cat(
        (
            smoothstep(torch.linspace(0, 1, fade_in)),
            torch.ones(total_size - fade_in - fade_out),
            smoothstep(torch.linspace(1, 0, fade_out)),
        )
    )


def restitch(tiled, H, W, overtile=1):
    _, C, _, tile_size = tiled.shape
    n_rows = round(np.floor(H / tile_size) + overtile)
    n_cols = round(np.floor(W / tile_size) + overtile)
    out = torch.zeros((1, C, H, W), device=tiled.device)
    rescale = torch.zeros_like(out)  # required to ensure rounding errors don't cause blending artifacts
    i = 0
    ys = torch.linspace(0, H - tile_size, n_rows).round().long()
    xs = torch.linspace(0, W - tile_size, n_cols).round().long()
    fade = tile_size - ys[1]
    for y in ys:
        wy = blend_weight1d(tile_size, fade_in=0 if y == 0 else fade, fade_out=0 if y == ys[-1] else fade).to(tiled)
        for x in xs:
            wx = blend_weight1d(tile_size, fade_in=0 if x == 0 else fade, fade_out=0 if x == xs[-1] else fade).to(tiled)
            weight = wy.reshape(1, 1, -1, 1) * wx.reshape(1, 1, 1, -1)
            out[..., y : y + tile_size, x : x + tile_size] += tiled[i] * weight
            rescale[..., y : y + tile_size, x : x + tile_size] += weight
            i += 1
    return out / rescale


def build_output_name(init=None, style=None, text=None):
    out_name = str(uuid4())[:6]
    if text is not None:
        out_name = f"{text.replace(' ','_')}_{out_name}"
    if style is not None:
        out_name = f"{Path(style).stem}_{out_name}"
    if init is not None:
        out_name = f"{Path(init).stem}_{out_name}"
    return out_name


def get_image_sizes(width, height, scales, scale_factor):
    shapes = []
    for scale in range(scales):
        divisor = scale_factor ** (scales - 1 - scale)
        shapes.append((round64(height / divisor), round64(width / divisor)))
    return shapes


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


if __name__ == "__main__":
    with torch.no_grad():
        W, H = 768, 768
        tile_size = 256
        num_images = 1
        scales = 2
        sf = 3
        timesteps = 50
        skips = [0.0, 0.5]
        text = sys.argv[2]
        init = sys.argv[1]
        style_img = None
        super_res_model = "SwinIR-M-DFO-GAN"
        match_hist = False
        sharpness_factor = 0
        stitch = True
        max_batch = 4
        diffusion_speed = "regular"
        diffusion_sampler = "plms"
        cfg_scale = 5
        clip_scale = 2500
        lpips_scale = 0
        style_scale = 0
        color_match_scale = 0
        device = "cuda"
        out_dir = "/home/hans/neurout/diffusion/"

        # initialize diffusion class
        # diffusion = GuidedDiffusion(
        #     [
        #         CLIPGrads(scale=clip_scale),
        #         LPIPSGrads(scale=lpips_scale),
        #         VGGGrads(scale=style_scale),
        #         ColorMatchGrads(scale=color_match_scale),
        #     ],
        #     sampler=diffusion_sampler,
        #     timesteps=timesteps,
        #     speed=diffusion_speed,
        # ).to(device)
        # diffusion = LatentDiffusion(sampler=diffusion_sampler, timesteps=timesteps).to(device)
        # diffusion = GLIDE(sampler=diffusion_sampler, timesteps=timesteps).to(device)
        diffusion = GLID3XL(sampler=diffusion_sampler, timesteps=timesteps).to(device)

        # calculate steps to start from (supports compound timestep respacing like '30,20,10')
        start_steps = get_start_steps(skips, diffusion)

        # calculate size of each scale
        shapes = get_image_sizes(W, H, scales, sf)

        if tile_size is None:
            tile_size = diffusion.model.image_size

        for b in range(num_images):

            # build output name based on inputs
            out_name = build_output_name(init, style_img, text)[:222]

            # initialize image
            img = initialize_image(init, shapes[0])
            content = img.clone()

            # maybe apply style image's color histogram to init image
            if match_hist and style_img is not None:
                img = match_histogram(img, StylePrompt(path=style_img).img)

            for scale, start_step in enumerate(start_steps):

                if scale != 0:
                    save(img, f"{out_dir}/{out_name}_{scale}.png")

                    # maybe upsample image with super-resolution model
                    if super_res_model:
                        img = upscale_image(img.add(1).div(2), model_name=super_res_model).mul(2).sub(1)

                    # resize image for next scale
                    img = resize(img, out_shape=shapes[scale], interp_method=lanczos3).cpu()

                print(f"Current size: {shapes[scale][1]}x{shapes[scale][0]}")

                # if the image is larger than diffuison model's size, chop it into tiles
                needs_stitching = stitch and min(shapes[scale]) > tile_size
                if needs_stitching:
                    img = destitch(img, tile_size=tile_size)

                # initialize prompts for diffusion
                prompts = [ContentPrompt(img=content).to(img)]
                if text is not None:
                    prompts.append(TextPrompt(text, weight=cfg_scale))
                if style_img is not None:
                    prompts.append(StylePrompt(path=style_img, size=shapes[scale]))

                # run diffusion sampling (in multiple batches if necessary)
                if img.shape[0] > max_batch:
                    img = [
                        diffusion.sample(
                            im_batch.to(device),
                            prompts=prompts,
                            start_step=start_step,
                            n_steps=start_step,
                            verbose=False,
                        )
                        for im_batch in tqdm(img.split(max_batch))
                    ]
                    img = torch.cat(img)
                else:
                    img = diffusion.sample(img.to(device), prompts=prompts, start_step=start_step, n_steps=start_step)

                # reassemble image tiles to final image
                if needs_stitching:
                    img = restitch(img, *shapes[scale])

                # maybe sharpen image
                if sharpness_factor:
                    img = adjust_sharpness(img.add(1).div(2), sharpness_factor=sharpness_factor).mul(2).sub(1)

            save(img, f"{out_dir}/{out_name}.png")
