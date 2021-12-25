import argparse
import os
import sys
from typing import  List

import torch
from PIL import Image

from maua_utils import download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.inference_mode()
def realesrgan(images: List[Image.Image]):
    sys.path.append("submodules/Real-ESRGAN")

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if not os.path.exists("modelzoo/RealESRGAN_x4plus.pth"):
        download(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "modelzoo/RealESRGAN_x4plus.pth",
        )

    upsampler = RealESRGANer(
        scale=4,
        model_path="modelzoo/RealESRGAN_x4plus.pth",
        model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).eval(),
        tile=0,
        half=True,
    )

    for img in images:
        yield upsampler.enhance(img, outscale=4)[0]

