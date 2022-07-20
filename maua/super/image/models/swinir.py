import os
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from ....ops.tensor import load_image
from ....utility import download

URLS = {
    "L-DFOWMFC-GAN": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
    "L-DFOWMFC-PSNR": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth",
    "M-DFO-GAN": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
    "M-DFO-PSNR": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth",
}


def load_model(model_name="L-DFOWMFC-GAN", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../../../submodules/SwinIR")
    from ....submodules.SwinIR.models.network_swinir import SwinIR

    model = (
        SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=240,
            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="3conv",
        )
        if "L" in model_name
        else SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="1conv",
        )
    )
    checkpoint = f"modelzoo/SwinIR-{model_name}.pth"
    if not os.path.exists(checkpoint):
        download(URLS[model_name], checkpoint)
    pretrained_model = torch.load(checkpoint)
    model.load_state_dict(
        pretrained_model["params_ema"] if "params_ema" in pretrained_model.keys() else pretrained_model, strict=True
    )
    model = model.eval().to(device)
    model.device = device
    return model


@torch.inference_mode()
def upscale(images: List[Union[Tensor, Image.Image, Path, str]], model):
    window_size = 8
    for img in images:
        img_lq = load_image(img).float().to(model.device)

        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, : h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, : w_old + w_pad]

        output = model(img_lq)
        output = output[..., : h_old * 4, : w_old * 4]

        yield output.float()
