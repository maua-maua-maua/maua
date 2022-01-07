import argparse
import os
import sys
from typing import List

import torch
from PIL import Image

from maua.utility import download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def bsrgan(images: List[Image.Image], model_name: str = "BSRGAN"):
    sys.path.append("submodules/BSRGAN")
    from utils import utils_image as util

    from submodules.BSRGAN.models.network_rrdbnet import RRDBNet as BSR_RRDBNet

    sf = 4

    model_path = f"modelzoo/{model_name}.pth"
    if not os.path.exists(model_path):
        urls = {
            "modelzoo/BSRGAN.pth": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth",
            "modelzoo/RealSR_JPEG.pth": "https://github.com/cszn/KAIR/releases/download/v1.0/RealSR_JPEG.pth",
        }
        download(urls[model_path], model_path)

    torch.cuda.empty_cache()

    model = BSR_RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf).eval()

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()

    for img in images:
        img_L = util.uint2tensor4(img)
        img_L = img_L.to(device)
        img_E = model(img_L)
        img_E = util.tensor2uint(img_E)
        img_E = img_E[:, :, [2, 1, 0]]
        yield Image.fromarray(img_E)
