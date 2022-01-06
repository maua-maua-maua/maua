import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms.functional import resize, to_tensor

from maua_utils import download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def latentdiffusion(images: List[Image.Image]):
    sys.path.append("./submodules/latent_diffusion")
    sys.path.append("./submodules/VQGAN")
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.util import instantiate_from_config, ismap
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
    model = model.cuda().eval()

    up_f = 4

    for img in images:
        c = torch.unsqueeze(to_tensor(img), 0)
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
        sample = sample.clamp(-1, 1).add(1).div(2).mul(255).squeeze().permute(1, 2, 0).cpu()
        yield Image.fromarray(sample.numpy().round().astype(np.uint8))
