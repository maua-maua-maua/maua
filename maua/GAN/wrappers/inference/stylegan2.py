# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Hans Brouwer for Maua
#
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .ops import (
    activation_funcs,
    bias_act,
    conv2d_resample,
    modulated_conv2d,
    normalize_2nd_moment,
    setup_filter,
    upsample2d,
)


class FullyConnectedLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,  # Number of input features.
        out_features: int,  # Number of output features.
        bias: bool = True,  # Apply additive bias before the activation function?
        activation: str = "linear",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 1.0,  # Learning rate multiplier.
        bias_init: float = 0.0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: Tensor):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and not self.bias_gain == 1.0:
            b = b * self.bias_gain

        if self.activation == "linear":
            x = torch.nn.functional.linear(x, w, b)
        else:
            x = bias_act(torch.nn.functional.linear(x, w.T, None), b, act=self.activation)
        return x


class Conv2dLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  # Number of input channels.
        out_channels: int,  # Number of output channels.
        kernel_size: int,  # Width and height of the convolution kernel.
        bias: bool = True,  # Apply additive bias before the activation function?
        activation: str = "linear",  # Activation function: 'relu', 'lrelu', etc.
        up: int = 1,  # Integer upsampling factor.
        down: int = 1,  # Integer downsampling factor.
        resample_filter: List[int] = [1, 3, 3, 1],  # Low-pass filter to apply when resampling activations
        conv_clamp: Optional[float] = None,  # Clamp the output to +-X, None = disable clamping.
        trainable: bool = True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = activation_funcs[activation]["def_gain"]

        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer("weight", weight)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

    def forward(self, x: Tensor, gain: float = 1.0):
        w = self.weight * self.weight_gain
        b = self.bias if self.bias is not None else None
        x = conv2d_resample(
            x=x,
            w=w,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
        )
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim: int,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim: int,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim: int,  # Intermediate latent (W) dimensionality.
        num_ws: int,  # Number of intermediate latents to output, None = do not broadcast.
        num_layers: int = 8,  # Number of mapping layers.
        embed_features: Optional[int] = None,  # Label embedding dimensionality, None = same as w_dim.
        layer_features: Optional[int] = None,  # Number of intermediate features in the mapping layers
        activation: str = "lrelu",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta: float = 0.998,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        self.embed = None
        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)

        fcs = []
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            fcs.append(layer)
        self.fcs = torch.nn.ModuleList(fcs)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self,
        z: Tensor,
        c: Optional[Tensor],
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
    ):
        # Embed, normalize, and concat inputs.
        x = None
        if self.z_dim > 0:
            x = normalize_2nd_moment(z)
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c))
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for layer in self.fcs:
            x = layer(x)

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1)
            x = torch.cat([x for _ in range(self.num_ws)], dim=1)

        # Apply truncation.
        if truncation_psi != 1:
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)

        return x


class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  # Number of input channels.
        out_channels: int,  # Number of output channels.
        w_dim: int,  # Intermediate latent (W) dimensionality.
        resolution: int,  # Resolution of this layer.
        kernel_size: int = 3,  # Convolution kernel size.
        up: int = 1,  # Integer upsampling factor.
        use_noise: bool = True,  # Enable noise input?
        activation: str = "lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter: List[int] = [1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp: Optional[float] = None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = activation_funcs[activation]["def_gain"]

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
        self.noise_adjusted = False
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x: Tensor, w: Tensor, noise_mode: str = "const", gain: float = 1.0):
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == "random":
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device)
        if self.use_noise and noise_mode == "const":
            noise = self.noise_const

        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class ToRGBLayer(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, w_dim: int, kernel_size: int = 1, conv_clamp: Optional[float] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x: Tensor, w: Tensor):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False)
        x = bias_act(x, self.bias, clamp=self.conv_clamp)
        return x


class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,  # Number of input channels, 0 = first block.
        out_channels: int,  # Number of output channels.
        w_dim: int,  # Intermediate latent (W) dimensionality.
        resolution: int,  # Resolution of this block.
        img_channels: int,  # Number of output color channels.
        is_last: bool,  # Is this the last block?
        architecture: str = "skip",  # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter: List[int] = [1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp: int = 256.0,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16: bool = False,  # Use FP16 for this block?
        **layer_kwargs,  # Arguments for SynthesisLayer.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        self.const = None
        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        self.conv0 = None
        if in_channels != 0:
            self.conv0 = SynthesisLayer(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                up=2,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                **layer_kwargs,
            )
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            conv_clamp=conv_clamp,
            **layer_kwargs,
        )
        self.num_conv += 1

        if is_last or architecture == "skip":
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim, conv_clamp=conv_clamp)
            self.num_torgb += 1

        self.skip = None
        if in_channels != 0 and architecture == "resnet":
            self.skip = Conv2dLayer(
                in_channels, out_channels, kernel_size=1, bias=False, up=2, resample_filter=resample_filter
            )

    def forward(
        self,
        x: Optional[Tensor],
        img: Optional[Tensor],
        ws: Tensor,
        noise_mode: str = "const",
    ) -> Tuple[Tensor, Tensor]:
        w_idx = 0

        # Input.
        if self.in_channels == 0:
            x = self.const
            x = torch.stack([x for _ in range(ws.shape[0])])

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, ws[:, w_idx], noise_mode, gain=1.0)
            w_idx += 1
        elif self.architecture == "resnet":
            y = self.skip(x, gain=sqrt(0.5))
            x = self.conv0(x, ws[:, w_idx], noise_mode, gain=1.0)
            w_idx += 1
            x = self.conv1(x, ws[:, w_idx], noise_mode, gain=sqrt(0.5))
            w_idx += 1
            x = y + x
        else:
            x = self.conv0(x, ws[:, w_idx], noise_mode, gain=1.0)
            w_idx += 1
            x = self.conv1(x, ws[:, w_idx], noise_mode, gain=1.0)
            w_idx += 1

        # ToRGB.
        if img is not None:
            img = upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == "skip":
            y = self.torgb(x, ws[:, w_idx])
            w_idx += 1
            y = y.contiguous()
            img = (img + y) if img is not None else y
        if img is None:
            img = torch.empty(0)  # never reached

        return x, img


class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        w_dim: int,  # Intermediate latent (W) dimensionality.
        img_resolution: int,  # Output image resolution.
        img_channels: int,  # Number of color channels.
        channel_base: int = 32768,  # Overall multiplier for the number of channels.
        channel_max: int = 512,  # Maximum number of channels in any layer.
        num_fp16_res: int = 0,  # Use FP16 for the N highest resolutions.
        **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        bs = []
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = res >= fp16_resolution
            is_last = res == self.img_resolution
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                **block_kwargs,
            )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            bs.append(block)
        self.bs = torch.nn.ModuleList(bs)

    def forward(self, ws: Tensor, noise_mode: str = "const"):
        w_idx = 0
        x = img = None
        for block in self.bs:
            block_ws = ws.narrow(1, w_idx, block.num_conv + block.num_torgb)
            x, img = block(x, img, block_ws, noise_mode)
            w_idx += block.num_conv
        return img


class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim: int,  # Input latent (Z) dimensionality.
        c_dim: int,  # Conditioning label (C) dimensionality.
        w_dim: int,  # Intermediate latent (W) dimensionality.
        img_resolution: int,  # Output resolution.
        img_channels: int,  # Number of output color channels.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        **synthesis_kwargs,  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(
        self,
        z: Tensor,
        c: Optional[Tensor] = None,
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[float] = None,
        noise_mode="const",
    ):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, noise_mode)
        return img
