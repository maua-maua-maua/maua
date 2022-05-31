import os
import sys
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image
from torch import Tensor

from maua.ops.tensor import load_image
from maua.utility import download

URLS = {
    "x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "x4plus-anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "xsx4-animevideo": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx4.pth",
    "pbaylies-wikiart": "https://archive.org/download/hr-painting-upscaling/wikiart_g.pth",
    "pbaylies-hr-paintings": "https://archive.org/download/hr-painting-upscaling/hr-paintings_g.pth",
}


def load_model(model_name="pbaylies-hr-paintings", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    sys.path.append(os.path.dirname(__file__) + "/../../../submodules/RealESRGAN")

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    checkpoint = f"modelzoo/RealESRGAN_{model_name}.pth"
    if not os.path.exists(checkpoint):
        download(URLS[model_name], checkpoint)

    if model_name == "x4plus-anime":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    elif model_name == "xsx4-animevideo":
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    return RealESRGANer(scale=4, model_path=checkpoint, model=model.eval(), tile=0, half=True)


@torch.inference_mode()
def upscale(images: List[Union[Tensor, Image.Image, Path, str]], model):
    for img in images:
        input = load_image(img).squeeze().permute(1, 2, 0).mul(255).numpy()
        large = model.enhance(input)[0]
        large = torch.from_numpy(large).permute(2, 0, 1).unsqueeze(0).float().div(255)
        yield large
