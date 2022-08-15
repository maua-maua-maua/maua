import os
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import resize

from ....ops.io import load_image
from ....utility import download


def load_model(model_name="latent-diffusion", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    for file in [
        os.path.dirname(__file__) + "/../../../submodules/latent_diffusion/ldm/models/diffusion/ddim.py",
        os.path.dirname(__file__) + "/../../../submodules/latent_diffusion/ldm/models/diffusion/ddpm.py",
        os.path.dirname(__file__) + "/../../../submodules/latent_diffusion/ldm/modules/diffusionmodules/model.py",
        os.path.dirname(__file__) + "/../../../submodules/latent_diffusion/ldm/util.py",
    ]:
        with open(file, "r") as f:
            txt = (
                f.read()
                .replace("print", "None # print")
                .replace("None # None #", "None #")
                .replace("    self.z_shape, np.prod(self.z_shape)))", "#    self.z_shape, np.prod(self.z_shape)))")
            )
        with open(file, "w") as f:
            f.write(txt)

    sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../../../submodules/latent_diffusion")
    sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../../../submodules/VQGAN")
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.util import instantiate_from_config
    from omegaconf import OmegaConf

    path_conf = "modelzoo/latent-diffusion/superresolution_bsr_config_project.yaml"
    path_ckpt = "modelzoo/latent-diffusion/superresolution_bsr_checkpoint_last.ckpt"
    if not os.path.exists(path_conf):
        download("https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1", path_conf)
    if not os.path.exists(path_ckpt):
        download("https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1", path_ckpt)

    model = instantiate_from_config(OmegaConf.load(path_conf).model)
    sd = torch.load(path_ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model, DDIMSampler, device


@torch.inference_mode()
def upscale(images: List[Union[Tensor, Image.Image, Path, str]], model):
    model, DDIMSampler, device = model
    up_f = 4
    for img in images:
        c = load_image(img)
        example = dict(
            LR_image=rearrange(c, "1 c h w -> 1 h w c").mul(2).add(-1).to(device),
            image=rearrange(resize(c, size=[up_f * c.size(2), up_f * c.size(3)], antialias=True), "1 c h w -> 1 h w c"),
        )
        height, width = example["image"].shape[1:3]
        if height >= 128 and width >= 128:
            model.split_input_params = {
                "ks": (128, 128),
                "stride": (64, 64),
                "vqf": 4,
                "patch_distributed_vq": True,
                "tie_braker": False,
                "clip_max_weight": 0.5,
                "clip_min_weight": 0.01,
                "clip_max_tie_weight": 0.5,
                "clip_min_tie_weight": 0.01,
            }
        else:
            if hasattr(model, "split_input_params"):
                delattr(model, "split_input_params")

        z, c, _, _, _ = model.get_input(
            example,
            model.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=not (hasattr(model, "split_input_params") and model.cond_stage_key == "coordinates_bbox"),
            return_original_cond=True,
        )
        with model.ema_scope("Plotting"):
            sample, intermediates = DDIMSampler(model).sample(
                100, batch_size=z.shape[0], shape=z.shape[1:], conditioning=c, eta=1.0, verbose=False
            )
        sample = model.decode_first_stage(sample)
        sample = sample.clamp(-1, 1).add(1).div(2)
        yield sample.float()
