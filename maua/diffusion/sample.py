from pathlib import Path
from uuid import uuid4

import torch
from numpy import linspace
from PIL import Image, ImageOps
from resize_right import resize
from resize_right.interp_methods import lanczos3
from torchvision.transforms.functional import adjust_sharpness, autocontrast, to_pil_image, to_tensor

from ..ops.loss import range_loss, tv_loss
from .conditioning import CLIPGrads, ContentPrompt, LossGrads, LPIPSGrads, TextPrompt
from .wrappers.guided import GuidedDiffusion

# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869


def interp(t):
    return 3 * t**2 - 2 * t**3


def perlin(width, height, scale=10, device="cuda"):
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


if __name__ == "__main__":
    with torch.no_grad():
        W, H = 512, 384
        num_images = 32
        scales = 1
        sf = 2
        timesteps = 100
        start_skip, end_skip = 0.1, 0.5
        text = "very very very beautiful optimistic solarpunk eco painting science fiction sci-fi futuristic cyberpunk digital art trending on ArtStation"
        init = None

        diffusion = GuidedDiffusion(
            [CLIPGrads(scale=8000)],  # , LossGrads(tv_loss, scale=60), LossGrads(range_loss, scale=75)],
            sampler="plms",
            timesteps=timesteps,
        )

        for b in range(num_images):
            out_name = f"{text.replace(' ','_')}_{str(uuid4())[:6]}"
            if init is not None:
                out_name = f"{Path(init).stem}_{out_name}"

            shape = round64(H / sf ** (scales - 1)), round64(W / sf ** (scales - 1))

            if init == "randn":
                img = torch.randn((1, 3, *shape))
            elif init is not None:
                img = to_tensor(Image.open(init).convert("RGB")).unsqueeze(0).mul(2).sub(1)
            else:
                img = (
                    resize(create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False), out_shape=shape)
                    + resize(create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True), out_shape=shape)
                    - 1
                ).unsqueeze(0)
            img = img.cuda(non_blocking=True)

            for scale, skip in enumerate(linspace(start_skip, end_skip, scales)):

                if scale != 0:
                    shape = round64(H / sf ** (scales - 1 - scale)), round64(W / sf ** (scales - 1 - scale))
                    img = resize(img, out_shape=shape, interp_method=lanczos3)
                    to_pil_image(img.squeeze().add(1).div(2).clamp(0, 1)).save(f"output/{out_name}_{scale}.png")

                print(img.shape[3], img.shape[2])

                steps = round(timesteps * (1 - skip))
                img = diffusion.sample(img, prompts=[TextPrompt(text)], start_step=steps, n_steps=steps)
                img = adjust_sharpness(img.add(1).div(2), sharpness_factor=2).mul(2).sub(1)

            to_pil_image(img.squeeze().add(1).div(2).clamp(0, 1)).save(f"output/{out_name}.png")
