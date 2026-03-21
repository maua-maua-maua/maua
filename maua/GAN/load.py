import os
import sys
import traceback
from copy import deepcopy
from functools import partial

import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/nv/")

import re

import numpy as np

from maua.GAN.nv import dnnlib, legacy
from maua.GAN.nv.networks import stylegan2 as stylegan2_train
from maua.GAN.nv.networks import stylegan3
from maua.GAN.wrappers.inference import stylegan2 as stylegan2_inference


def convert_to_rgb(state_ros, state_nv, ros_name, nv_name):
    state_ros[f"{ros_name}.conv.weight"] = state_nv[f"{nv_name}.torgb.weight"].unsqueeze(0)
    state_ros[f"{ros_name}.bias"] = state_nv[f"{nv_name}.torgb.bias"].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    state_ros[f"{ros_name}.conv.modulation.weight"] = state_nv[f"{nv_name}.torgb.affine.weight"]
    state_ros[f"{ros_name}.conv.modulation.bias"] = state_nv[f"{nv_name}.torgb.affine.bias"]


def convert_conv(state_ros, state_nv, ros_name, nv_name):
    state_ros[f"{ros_name}.conv.weight"] = state_nv[f"{nv_name}.weight"].unsqueeze(0)
    state_ros[f"{ros_name}.activate.bias"] = state_nv[f"{nv_name}.bias"]
    state_ros[f"{ros_name}.conv.modulation.weight"] = state_nv[f"{nv_name}.affine.weight"]
    state_ros[f"{ros_name}.conv.modulation.bias"] = state_nv[f"{nv_name}.affine.bias"]
    state_ros[f"{ros_name}.noise.weight"] = state_nv[f"{nv_name}.noise_strength"].unsqueeze(0)


def convert_blur_kernel(state_ros, state_nv, level):
    """Not quite sure why there is a factor of 4 here"""
    # They are all the same
    state_ros[f"convs.{2 * level}.conv.blur.kernel"] = 4 * state_nv["synthesis.b4.resample_filter"]
    state_ros[f"to_rgbs.{level}.upsample.kernel"] = 4 * state_nv["synthesis.b4.resample_filter"]


def determine_config(state_nv):
    mapping_names = [name for name in state_nv.keys() if "mapping.fc" in name]
    sythesis_names = [name for name in state_nv.keys() if "synthesis.b" in name]

    n_mapping = max([int(re.findall(r"(\d+)", n)[0]) for n in mapping_names]) + 1
    resolution = max([int(re.findall(r"(\d+)", n)[0]) for n in sythesis_names])
    n_layers = np.log(resolution / 2) / np.log(2)

    return n_mapping, n_layers


def ada2ros(state_nv):
    """Adapted from https://github.com/dvschultz/stylegan2-ada-pytorch/blob/main/export_weights.py"""
    n_mapping, n_layers = determine_config(state_nv)

    state_ros = {}

    for i in range(n_mapping):
        state_ros[f"style.{i + 1}.weight"] = state_nv[f"mapping.fc{i}.weight"]
        state_ros[f"style.{i + 1}.bias"] = state_nv[f"mapping.fc{i}.bias"]

    for i in range(int(n_layers)):
        if i > 0:
            for conv_level in range(2):
                convert_conv(
                    state_ros, state_nv, f"convs.{2 * i - 2 + conv_level}", f"synthesis.b{4 * (2**i)}.conv{conv_level}"
                )
                state_ros[f"noises.noise_{2 * i - 1 + conv_level}"] = (
                    state_nv[f"synthesis.b{4 * (2**i)}.conv{conv_level}.noise_const"].unsqueeze(0).unsqueeze(0)
                )

            convert_to_rgb(state_ros, state_nv, f"to_rgbs.{i - 1}", f"synthesis.b{4 * (2**i)}")
            convert_blur_kernel(state_ros, state_nv, i - 1)

        else:
            state_ros["input.input"] = state_nv[f"synthesis.b{4 * (2**i)}.const"].unsqueeze(0)
            convert_conv(state_ros, state_nv, "conv1", f"synthesis.b{4 * (2**i)}.conv1")
            state_ros[f"noises.noise_{2 * i}"] = (
                state_nv[f"synthesis.b{4 * (2**i)}.conv1.noise_const"].unsqueeze(0).unsqueeze(0)
            )
            convert_to_rgb(state_ros, state_nv, "to_rgb1", f"synthesis.b{4 * (2**i)}")

    # https://github.com/yuval-alaluf/restyle-encoder/issues/1#issuecomment-828354736
    latent_avg = state_nv["mapping.w_avg"]
    state_dict = {"g_ema": state_ros, "latent_avg": latent_avg}
    return state_dict


def load_rosinality2ada(path, blur_scale=4.0, for_inference=False):
    state_dict = torch.load(path)
    state_ros = state_dict
    if "g_ema" in state_dict:
        state_ros = state_dict["g_ema"]
    state_nv = {}

    nv_key = "bs.0" if for_inference else "b4"
    if tuple(state_ros["input.input"].shape) != (1,):
        state_nv[f"synthesis.{nv_key}.const"] = state_ros["input.input"].squeeze(0)
    else:
        state_nv[f"synthesis.{nv_key}.const.affine.weight"] = state_ros["input.linear.weight"].squeeze(0)
        state_nv[f"synthesis.{nv_key}.const.affine.bias"] = state_ros["input.linear.bias"].squeeze(0)

    state_nv[f"synthesis.{nv_key}.conv1.noise_const"] = state_ros["noises.noise_0"].squeeze(0).squeeze(0)

    state_nv[f"synthesis.{nv_key}.conv1.weight"] = state_ros["conv1.conv.weight"].squeeze(0)
    state_nv[f"synthesis.{nv_key}.conv1.bias"] = state_ros["conv1.activate.bias"]
    state_nv[f"synthesis.{nv_key}.conv1.affine.weight"] = state_ros["conv1.conv.modulation.weight"]
    state_nv[f"synthesis.{nv_key}.conv1.affine.bias"] = state_ros["conv1.conv.modulation.bias"]
    if not for_inference:
        state_nv[f"synthesis.{nv_key}.conv1.noise_strength"] = state_ros["conv1.noise.weight"].squeeze(0)

    state_nv[f"synthesis.{nv_key}.torgb.weight"] = state_ros["to_rgb1.conv.weight"].squeeze(0)
    state_nv[f"synthesis.{nv_key}.torgb.bias"] = state_ros["to_rgb1.bias"].squeeze(-1).squeeze(-1).squeeze(0)
    state_nv[f"synthesis.{nv_key}.torgb.affine.weight"] = state_ros["to_rgb1.conv.modulation.weight"]
    state_nv[f"synthesis.{nv_key}.torgb.affine.bias"] = state_ros["to_rgb1.conv.modulation.bias"]
    state_nv[f"synthesis.{nv_key}.resample_filter"] = state_ros["convs.0.conv.blur.kernel"] / blur_scale
    state_nv[f"synthesis.{nv_key}.conv1.resample_filter"] = state_ros["convs.0.conv.blur.kernel"] / blur_scale

    max_res, num_map = 4, 1
    for key, val in state_ros.items():
        if key.startswith("style"):
            _, num, weight_or_bias = key.split(".")
            nv_key = (
                f"mapping.fcs.{int(num) - 1}.{weight_or_bias}"
                if for_inference
                else f"mapping.fc{int(num) - 1}.{weight_or_bias}"
            )
            state_nv[nv_key] = val

            num_map = max(num_map, int(num))

        if key.startswith("noises"):
            n = int(key.split("_")[1])
            r = 2 ** (3 + (n - 1) // 2)
            nv_block = f"synthesis.bs.{(n - 1) // 2 + 1}" if for_inference else f"synthesis.b{r}"
            state_nv[f"{nv_block}.conv{(n - 1) % 2}.noise_const"] = state_ros[f"noises.noise_{n}"].squeeze(0).squeeze(0)

        if key.startswith("convs"):
            n = int(key.split(".")[1])
            r = 2 ** (3 + n // 2)
            nv_block = f"synthesis.bs.{(n // 2) + 1}" if for_inference else f"synthesis.b{r}"
            ros_name = ".".join(key.split(".")[2:])

            if ros_name == "conv.weight":
                state_nv[f"{nv_block}.conv{n % 2}.weight"] = state_ros[f"convs.{n}.conv.weight"].squeeze(0)
            elif ros_name == "activate.bias":
                state_nv[f"{nv_block}.conv{n % 2}.bias"] = state_ros[f"convs.{n}.activate.bias"]
            elif ros_name == "conv.modulation.weight":
                state_nv[f"{nv_block}.conv{n % 2}.affine.weight"] = state_ros[f"convs.{n}.conv.modulation.weight"]
            elif ros_name == "conv.modulation.bias":
                state_nv[f"{nv_block}.conv{n % 2}.affine.bias"] = state_ros[f"convs.{n}.conv.modulation.bias"]
            elif ros_name == "noise.weight" and not for_inference:
                state_nv[f"{nv_block}.conv{n % 2}.noise_strength"] = state_ros[f"convs.{n}.noise.weight"].squeeze(0)
            elif ros_name == "conv.blur.kernel":
                state_nv[f"{nv_block}.conv0.resample_filter"] = state_ros[f"convs.{n}.conv.blur.kernel"] / blur_scale
                state_nv[f"{nv_block}.conv1.resample_filter"] = state_ros[f"convs.{n}.conv.blur.kernel"] / blur_scale
            else:
                raise Exception(f"Key {key} not recognized!")

            max_res = max(max_res, r)

        if key.startswith("to_rgbs"):
            n = int(key.split(".")[1])
            r = 2 ** (3 + n)
            nv_block = f"synthesis.bs.{n + 1}" if for_inference else f"synthesis.b{r}"
            ros_name = ".".join(key.split(".")[2:])

            if ros_name == "conv.weight":
                state_nv[f"{nv_block}.torgb.weight"] = state_ros[f"to_rgbs.{n}.conv.weight"].squeeze(0)
            elif ros_name == "bias":
                state_nv[f"{nv_block}.torgb.bias"] = state_ros[f"to_rgbs.{n}.bias"].squeeze(-1).squeeze(-1).squeeze(0)
            elif ros_name == "conv.modulation.weight":
                state_nv[f"{nv_block}.torgb.affine.weight"] = state_ros[f"to_rgbs.{n}.conv.modulation.weight"]
            elif ros_name == "conv.modulation.bias":
                state_nv[f"{nv_block}.torgb.affine.bias"] = state_ros[f"to_rgbs.{n}.conv.modulation.bias"]
            elif ros_name == "upsample.kernel":
                state_nv[f"{nv_block}.resample_filter"] = state_ros[f"to_rgbs.{n}.upsample.kernel"] / blur_scale
            else:
                raise Exception(f"Key {key} not recognized!")

    if "latent_avg" in state_dict:
        state_nv["mapping.w_avg"] = state_dict["latent_avg"]
    else:
        state_nv["mapping.w_avg"] = torch.zeros(512)  # TODO

    z_dim = 512  # TODO
    w_dim = 512  # TODO
    c_dim = 0  # TODO
    chnls = 3  # TODO

    G = (stylegan2_inference if for_inference else stylegan2_train).Generator(
        z_dim,
        c_dim,
        w_dim,
        max_res,
        chnls,
        mapping_kwargs=dict(num_layers=num_map),  # , use_const=use_const
    )
    G.load_state_dict(state_nv)

    return G


def load_nvidia(path, for_inference=None):
    with dnnlib.util.open_url(path) as f:
        G_persistence = legacy.load_network_pkl(f)["G_ema"]

    # create new Generator class to avoid the uninformative errors from NVIDIA's persistence system
    try:
        G = stylegan3.Generator(
            G_persistence.mapping.z_dim,
            G_persistence.mapping.c_dim,
            G_persistence.mapping.w_dim,
            G_persistence.img_resolution,
            G_persistence.img_channels,
            mapping_kwargs=dict(num_layers=G_persistence.mapping.num_layers),
        )
        G.load_state_dict(G_persistence.state_dict())
        try_stylegan2 = False
    except:
        try_stylegan2 = True

    if try_stylegan2:
        try:
            G = (stylegan2_inference if for_inference else stylegan2_train).Generator(
                G_persistence.mapping.z_dim,
                G_persistence.mapping.c_dim,
                G_persistence.mapping.w_dim,
                G_persistence.img_resolution,
                G_persistence.img_channels,
                mapping_kwargs=dict(num_layers=G_persistence.mapping.num_layers),
            )
            G.load_state_dict(G_persistence.state_dict())
        except:
            G = deepcopy(G_persistence)

    del G_persistence
    return G


def load_nvidia_pt(
    path, z_dim=512, c_dim=0, w_dim=512, img_resolution=1024, img_channels=3, map_layers=8, for_inference=False
):
    state_dict = torch.load(path)["G_ema"]

    # create new Generator class to avoid the uninformative errors from NVIDIA's persistence system
    try:
        G = stylegan3.Generator(
            z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs=dict(num_layers=map_layers)
        )
        G.load_state_dict(state_dict)
        try_stylegan2 = False
    except:
        try_stylegan2 = True

    if try_stylegan2:
        G = (stylegan2_inference if for_inference else stylegan2_train).Generator(
            z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs=dict(num_layers=map_layers)
        )
        G.load_state_dict(state_dict)

    del state_dict
    return G


def load_network(path, for_inference=False):
    errors = {}

    for name, loader in [
        ("NVIDIA StyleGAN3 loader", load_nvidia),
        ("NVIDIA non-persistence loader", load_nvidia_pt),
        ("Rosinality StyleGAN2 to ADA-PT converter", load_rosinality2ada),
        ("Rosinality StyleGAN2 to Inference converter", partial(load_rosinality2ada, for_inference=True)),
    ]:
        try:
            return loader(path, for_inference=for_inference)
        except:
            errors[name] = traceback.format_exc()

    error_str = "\n".join([f"\n{k}:\n{e}\n" for k, e in errors.items()])
    raise Exception(f"Error loading checkpoint! None of the converters succeeded:\n{error_str}")
