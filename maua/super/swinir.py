import os
import sys
from typing import List

import numpy as np
import torch
from maua.utility import download
from PIL import Image
from torchvision.transforms.functional import to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def swinir(images: List[Image.Image]):
    sys.path.append("submodules/SwinIR")
    from submodules.SwinIR.models.network_swinir import SwinIR

    model = SwinIR(
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
    if not os.path.exists("modelzoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"):
        download(
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
            "modelzoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
        )
    pretrained_model = torch.load("modelzoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth")
    model.load_state_dict(
        pretrained_model["params_ema"] if "params_ema" in pretrained_model.keys() else pretrained_model, strict=True
    )
    model = model.eval().to(device)

    window_size = 8
    for img in to_tensor(images):
        img_lq = img.float().unsqueeze(0).to(device)

        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, : h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, : w_old + w_pad]
        output = model(img_lq)
        output = output[..., : h_old * 4, : w_old * 4]
        output = (output.data.squeeze().float().cpu().clamp_(0, 1).numpy() * 255.0).round().astype(np.uint8)

        yield Image.fromarray(output)
