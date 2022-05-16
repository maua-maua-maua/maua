import random
import sys
from glob import glob
from pathlib import Path
from uuid import uuid4

import torch
from maua.super.image import upscale
from numpy import linspace
from PIL import Image
from resize_right import resize
from torchvision.transforms.functional import to_pil_image, to_tensor

from ..ops.loss import saturation_loss
from .conditioning import CLIPGrads, ContentPrompt, LPIPSGrads, LossGrads, TextPrompt
from .wrappers.guided import GuidedDiffusion


def round64(x):
    return round(x / 64) * 64


with torch.no_grad():
    H, W = 2048, 1408
    num_images = 32
    resolutions = 2
    timesteps = 100
    start_skip, end_skip = 0, 0.9
    text = "Detailed geometric vector art on a dark black background, cyberpunk noir sci-fi mechanical butterfly wings made of crystal"

    sf = (resolutions - 1) * 4
    init_path = None

    for b in range(num_images):
        diffusion = GuidedDiffusion(
            [
                CLIPGrads(scale=35_000),
                LPIPSGrads(scale=15_000),
                LossGrads(saturation_loss, scale=6_000),
            ],
            sampler="plms",
            timesteps=timesteps,
        )

        if start_skip > 0:
            init_path = random.choice(glob(sys.argv[1] + "/*.png"))
            img = to_tensor(Image.open(init_path).convert("RGB")).mul(2).sub(1).unsqueeze(0).cuda(non_blocking=True)
            img = resize(img, out_shape=(round64(H / sf), round64(W / sf)))
        else:
            img = torch.randn((1, 3, round64(H / sf), round64(W / sf))).cuda(non_blocking=True)

        for s, skip in enumerate(linspace(start_skip, end_skip, resolutions)):
            steps = round(timesteps * (1 - skip))
            if s != 0:
                img = next(iter(upscale([img.add(1).div(2)], model_name="SwinIR-L-DFOWMFC-GAN"))).mul(2).sub(1)
            if s == resolutions - 1:
                img = resize(img, out_shape=(H, W))
            img = diffusion.sample(
                img,
                prompts=[TextPrompt(text)] + ([ContentPrompt(img)] if s != 0 else []),
                start_step=steps,
                n_steps=steps,
            )

        out_name = f"{text.replace(' ','_')}_{str(uuid4())[:6]}"
        if init_path is not None:
            out_name = f"{Path(init_path).stem}_{out_name}"
        to_pil_image(img.squeeze().add(1).div(2).clamp(0, 1)).save(f"output/{out_name}.png")
