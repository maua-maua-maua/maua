import os
import sys
from glob import glob

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from maua.ops.loss import clamp_with_grad, replace_grad
from maua.utility import download

from . import Parameterization

# replace checkpoint path to avoid the weird path that gets created by default as well as a bunch of prints
for file in [
    "maua/submodules/VQGAN/taming/models/vqgan.py",
    "maua/submodules/VQGAN/taming/modules/losses/lpips.py",
    "maua/submodules/VQGAN/taming/modules/losses/vqperceptual.py",
    "maua/submodules/VQGAN/taming/modules/diffusionmodules/model.py",
]:
    with open(file, "r") as f:
        t = (
            f.read()
            .replace("print", "None # print")
            .replace("None # None #", "None #")
            .replace("    self.z_shape, np.prod(self.z_shape)))", "#    self.z_shape, np.prod(self.z_shape)))")
            .replace('get_ckpt_path(name, "taming/modules/autoencoder/', 'get_ckpt_path(name, "modelzoo/')
        )
    with open(file, "w") as f:
        f.write(t)

sys.path.append("maua/submodules/VQGAN")

from taming.models import cond_transformer, vqgan


def maybe_download_vqgan(model_dir):
    # fmt: off
    if model_dir == "imagenet_1024":
        config_path, checkpoint_path = "modelzoo/vqgan_imagenet_f16_1024.yaml", "modelzoo/vqgan_imagenet_f16_1024.ckpt"
        if not os.path.exists(checkpoint_path):
            download("http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.yaml", config_path)
            download("http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.ckpt", checkpoint_path)
    elif model_dir == "imagenet_16384":
        config_path, checkpoint_path = "modelzoo/vqgan_imagenet_f16_16384.yaml", "modelzoo/vqgan_imagenet_f16_16384.ckpt"
        if not os.path.exists(checkpoint_path):
            download("http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.yaml", config_path)
            download("http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.ckpt", checkpoint_path)
    elif model_dir == "coco":
        config_path, checkpoint_path = "modelzoo/coco.yaml", "modelzoo/coco.ckpt"
        if not os.path.exists(checkpoint_path):
            download("https://dl.nmkd.de/ai/clip/coco/coco.yaml", config_path)
            download("https://dl.nmkd.de/ai/clip/coco/coco.ckpt", checkpoint_path)
    elif model_dir == "faceshq":
        config_path, checkpoint_path = "modelzoo/faceshq.yaml", "modelzoo/faceshq.ckpt"
        if not os.path.exists(checkpoint_path):
            download("https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT", config_path)
            download("https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt", checkpoint_path)
    elif model_dir == "wikiart_1024":
        config_path, checkpoint_path = "modelzoo/wikiart_1024.yaml", "modelzoo/wikiart_1024.ckpt"
        if not os.path.exists(checkpoint_path):
            download("http://mirror.io.community/blob/vqgan/wikiart.yaml", config_path)
            download("http://mirror.io.community/blob/vqgan/wikiart.ckpt", checkpoint_path)
    elif model_dir == "wikiart_16384":
        config_path, checkpoint_path = "modelzoo/wikiart_16384.yaml", "modelzoo/wikiart_16384.ckpt"
        if not os.path.exists(checkpoint_path):
            download("http://mirror.io.community/blob/vqgan/wikiart_16384.yaml", config_path)
            download("http://mirror.io.community/blob/vqgan/wikiart_16384.ckpt", checkpoint_path)
    elif model_dir == "sflckr":
        config_path, checkpoint_path = "modelzoo/sflckr.yaml", "modelzoo/sflckr.ckpt"
        if not os.path.exists(checkpoint_path):
            download("https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1", config_path)
            download("https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1", checkpoint_path)
    else:
        config_path = sorted(glob(model_dir + "/*.yaml"), reverse=True)[0]
        checkpoint_path = sorted(glob(model_dir + "/*.ckpt"), reverse=True)[0]
    # fmt: on
    return config_path, checkpoint_path


def load_vqgan_model(model_dir):
    config_path, checkpoint_path = maybe_download_vqgan(model_dir)
    config = OmegaConf.load(config_path)
    if config.model.target == "taming.models.vqgan.VQModel":
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f"unknown vqgan type: {config.model.target}")
    del model.loss
    return model


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class VQGAN(Parameterization):
    def __init__(
        self,
        height,
        width,
        tensor=None,
        vqgan_model="imagenet_16384",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ema=False,
    ):
        if tensor is None:
            tensor = torch.empty(1, 3, height, width, device=device).uniform_()

        model = load_vqgan_model(vqgan_model).to(tensor.device)
        tensor = model.encode(tensor.clamp(0, 1) * 2 - 1)[0]

        Parameterization.__init__(self, tensor.shape[2], tensor.shape[3], tensor, ema)

        self.model = model
        self.codebook = self.model.quantize.embedding.weight

    def decode(self, tensor=None):
        if tensor is None:
            tensor = self.tensor
        tens_q = vector_quantize(self.tensor.movedim(1, 3), self.codebook).movedim(3, 1)
        return clamp_with_grad(self.model.decode(tens_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def encode(self, tensor):
        self.tensor.set_(self.model.encode(tensor.clamp(0, 1) * 2 - 1)[0].data)

    def forward(self):
        return self.decode()
