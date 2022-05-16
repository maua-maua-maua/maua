import os
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from maua.ops.tensor import load_image
from maua.utility import download

URLS = {
    "BSRGAN": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth",
    "RealSR": "https://github.com/cszn/KAIR/releases/download/v1.0/RealSR_JPEG.pth",
}


def load_model(model_name="BSRGAN", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    with open("maua/submodules/BSRGAN/models/network_rrdbnet.py", "r") as f:
        txt = f.read().replace("print", "None # print").replace("None # None #", "None #")
    with open("maua/submodules/BSRGAN/models/network_rrdbnet.py", "w") as f:
        f.write(txt)

    sys.path.append("maua/submodules/BSRGAN")
    from maua.submodules.BSRGAN.models.network_rrdbnet import RRDBNet

    checkpoint = f"modelzoo/{model_name}.pth"
    if not os.path.exists(checkpoint):
        download(URLS[model_name], checkpoint)

    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4).eval()
    model.load_state_dict(torch.load(checkpoint), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    model.device = device

    return model


@torch.inference_mode()
def upscale(images: List[Union[Tensor, Image.Image, Path, str]], model):
    for img in images:
        img_L = load_image(img).to(model.device)
        img_E = model(img_L)
        yield img_E.float()
