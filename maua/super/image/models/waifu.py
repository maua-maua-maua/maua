import os
import sys
import zipfile

import numpy as np
import py7zr
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

from maua.ops.tensor import load_image

sys.path.append("maua/submodules/waifu2x")

from Models import CARN_V2, UpConv_7, Vgg_7, network_to_half  # , DCSCN


def load_model(model_name="upconv-anime-1", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    if "upconv" in model_name:
        upconv, which, noise = model_name.split("-")
        base_path = "maua/submodules/waifu2x/model_check_points/Upconv_7/"
        if not os.path.exists(f"modelzoo/{which}"):
            with py7zr.SevenZipFile(f"{base_path}/{which}.7z", mode="r") as z:
                z.extractall(path="modelzoo")
        model = UpConv_7()
        model.load_pre_train_weights(json_file=f"modelzoo/{which}/{noise}_scale2.0x_model.json")

    # TODO seems that the scale2x models don't actually upscale?
    # elif "vgg" in model_name:
    #     vgg, which = model_name.split("-")
    #     base_path = "maua/submodules/waifu2x/model_check_points/vgg_7/"
    #     if not os.path.exists(f"modelzoo/{which}"):
    #         with py7zr.SevenZipFile(f"{base_path}/{which}.7z", mode="r") as z:
    #             z.extractall(path="modelzoo")
    #     model = Vgg_7()
    #     model.load_pre_train_weights(json_file=f"modelzoo/{which}/scale2.0x_model.json")

    elif model_name == "CARN":
        base_path = "maua/submodules/waifu2x/model_check_points/CRAN_V2/"
        if not os.path.exists(f"modelzoo/CARN_model_checkpoint.pt"):
            with zipfile.ZipFile(f"{base_path}/CRAN_V2_02_28_2019.zip", "r") as zip_ref:
                zip_ref.extractall("modelzoo")
        model = CARN_V2(
            color_channels=3,
            mid_channels=64,
            conv=nn.Conv2d,
            single_conv_size=3,
            single_conv_group=1,
            scale=2,
            activation=nn.LeakyReLU(0.1),
            SEBlock=True,
            repeat_blocks=3,
            atrous=(1, 1, 1),
        )
        model = network_to_half(model)
        model.load_state_dict(torch.load(f"modelzoo/CARN_model_checkpoint.pt"))

    # TODO checkpoint parameter restore and input shape errors
    # elif model_name == "DCSCN":
    #     model = DCSCN(
    #         color_channel=3,
    #         up_scale=2,
    #         feature_layers=12,
    #         first_feature_filters=196,
    #         last_feature_filters=48,
    #         reconstruction_filters=64,
    #         up_sampler_filters=32,
    #     )
    #     model.load_state_dict(
    #         torch.load("maua/submodules/waifu2x/model_check_points/DCSCN/DCSCN_weights_387epos_L12_noise_1.pt")
    #     )

    model = network_to_half(model).eval().to(device)
    model.device = device
    return model


def windowed_index(device: torch.device, height: int, width: int, scale: int, pad_size: int, seg_size: int):
    if height % seg_size < pad_size or width % seg_size < pad_size:
        seg_size += scale * pad_size

    ys = torch.arange(pad_size, height, seg_size, dtype=torch.long, device=device)
    xs = torch.arange(pad_size, width, seg_size, dtype=torch.long, device=device)
    ys, xs = torch.meshgrid(ys, xs, indexing="ij")
    idxs = torch.stack([ys.flatten(), xs.flatten()])

    winrange = torch.arange(-pad_size, pad_size + seg_size, dtype=torch.long, device=device)
    ywin, xwin = torch.meshgrid(winrange, winrange, indexing="ij")
    window = torch.stack((ywin, xwin))

    idxs = idxs[:, :, None, None] + window[:, None, :, :]

    return idxs[0].clamp(0, height - 1), idxs[1].clamp(0, width - 1)


def split(img: torch.Tensor, scale: int, pad_size: int, seg_size: int):
    img = nn.functional.pad(img, [pad_size] * 4, mode="replicate")
    _, _, height, width = img.size()
    ys, xs = windowed_index(img.device, height, width, scale, pad_size, seg_size)
    patch_box = img[:, :, ys, xs].squeeze().permute(1, 0, 2, 3)
    return patch_box, height, width


def merge(img: torch.Tensor, height: int, width: int, scale: int, pad_size: int, seg_size: int):
    ys, xs = windowed_index(img.device, height * scale, width * scale, scale, pad_size * scale, seg_size * scale)
    rem = pad_size * 2
    img = img[..., rem:-rem, rem:-rem].permute(1, 0, 2, 3).unsqueeze(0).float()
    ys = ys[..., rem:-rem, rem:-rem]
    xs = xs[..., rem:-rem, rem:-rem]
    out = torch.zeros((1, 3, height * scale, width * scale), device=img.device)
    out[:, :, ys, xs] = img
    return out[..., rem:-rem, rem:-rem]


def upscale(images, model, scale=4, pad_size=3, seg_size=64, batch_size=256):
    for img in images:
        img = load_image(img).to(model.device)
        for _ in range(round(np.log2(scale))):
            img_patches, h, w = split(img, 2, pad_size, seg_size)
            larger_patches = torch.cat([model(patches) for patches in torch.split(img_patches, batch_size)])
            img = merge(larger_patches, h, w, 2, pad_size, seg_size).clamp(0, 1)
        yield img.float()
