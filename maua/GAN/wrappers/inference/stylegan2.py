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
from torch.nn.functional import linear

from .ops import activation_funcs, bias_act, conv2d_resample, setup_filter, upsample2d


def normalize_2nd_moment(x, dim: int = 1, eps: float = 1e-8):
    return x / ((x * x).mean(dim=dim, keepdim=True) + eps).sqrt()


def modulated_conv2d(
    x: Tensor,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight: Tensor,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles: Tensor,  # Modulation coefficients of shape [batch_size, in_channels].
    noise: Optional[Tensor] = None,  # Optional noise tensor to add to the output activations.
    up: int = 1,  # Integer upsampling factor.
    down: int = 1,  # Integer downsampling factor.
    padding: int = 0,  # Padding with respect to the upsampled image.
    resample_filter: Optional[Tensor] = None,  # Low-pass filter to apply when resampling activations.
    demodulate: bool = True,  # Apply weight demodulation?
    fused_modconv: bool = True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = int(x.shape[0])
    _, in_channels, kh, kw = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / weight.norm(float("inf"), dim=[1, 2, 3], keepdim=True) / sqrt(in_channels * kh * kw))
        styles = styles / styles.norm(float("inf"), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = dcoefs = x  # pre-annotate for torch.jit https://github.com/pytorch/pytorch/issues/57717#issuecomment-833613833
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = 1 / ((w * w).sum(dim=[2, 3, 4]) + 1e-8).sqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.reshape(batch_size, -1, 1, 1)
        x = conv2d_resample(x=x, w=weight, f=resample_filter, up=up, down=down, padding=padding)
        if demodulate and noise is not None:
            x = torch.addcmul(noise, x, dcoefs.reshape(batch_size, -1, 1, 1))
        elif demodulate:
            x = x * dcoefs.reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise)
        return x

    # Execute as one fused op using grouped convolution.
    batch_size = int(x.shape[0])
    xh, xw = [int(z) for z in x.shape[2:]]
    x = x.reshape(1, -1, xh, xw)
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample(
        x=x,
        w=w,
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        groups=batch_size,
    )
    xh, xw = [int(z) for z in x.shape[2:]]
    x = x.reshape(batch_size, -1, xh, xw)
    if noise is not None:
        x = x.add_(noise)
    return x


class FullyConnectedLayer(torch.nn.Module):
    __constants__ = ["in_features", "out_features", "activation", "weight_gain", "bias_gain"]

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
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: Tensor):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None:
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            x = linear(x, w, b)
        else:
            x = bias_act(linear(x, w.T, None), b, act=self.activation)
        return x

    def extra_repr(self):
        return f"in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}"


class Conv2dLayer(torch.nn.Module):
    __constants__ = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "activation",
        "up",
        "down",
        "conv_clamp",
        "channels_last",
        "trainable",
    ]

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
        channels_last: bool = False,  # Expect the input to have memory_format=channels_last?
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
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = activation_funcs[activation]["def_gain"]

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).contiguous(
            memory_format=memory_format
        )
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

    def extra_repr(self):
        return " ".join(
            [
                f"in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},",
                f"up={self.up}, down={self.down}",
            ]
        )


class MappingNetwork(torch.nn.Module):
    __constants__ = ["z_dim", "c_dim", "w_dim", "num_ws", "num_layers", "w_avg_beta"]

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

        self.embed = torch.jit.annotate(Optional[FullyConnectedLayer], None)
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
        update_emas: bool = False,
    ):
        # Embed, normalize, and concat inputs.
        x = torch.jit.annotate(Optional[Tensor], None)
        if self.z_dim > 0:
            x = normalize_2nd_moment(z)
        if self.embed is not None:
            y = normalize_2nd_moment(self.embed(c))
            x = torch.cat([x, y], dim=1) if x is not None else y
        if x is None:
            x = z  # thanks torch.jit

        # Main layers.
        for layer in self.fcs:
            x = layer(x)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            x = torch.cat([x.unsqueeze(1) for _ in range(self.num_ws)], dim=1)

        # Apply truncation.
        if truncation_psi != 1:
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f"z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}"


class SynthesisLayer(torch.nn.Module):
    __constants__ = [
        "in_channels",
        "out_channels",
        "w_dim",
        "resolution",
        "up",
        "use_noise",
        "activation",
        "conv_clamp",
        "padding",
        "act_gain",
    ]

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
        channels_last: bool = False,  # Use channels_last format for the weights?
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
        self.noise_adjusted = torch.tensor(0)

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).contiguous(memory_format=memory_format)
        )
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x: Tensor, w: Tensor, noise_mode: str = "random", fused_modconv: bool = True, gain: float = 1.0):
        styles = self.affine(w)

        noise = torch.jit.annotate(Optional[Tensor], None)
        if self.use_noise and noise_mode == "random":
            noise = (
                torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
            )
        if self.use_noise and noise_mode == "const":
            noise = self.noise_const * self.noise_strength

        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            fused_modconv=fused_modconv,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return " ".join(
            [
                f"in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},",
                f"resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}",
            ]
        )


class ToRGBLayer(torch.nn.Module):
    __constants__ = ["in_channels", "out_channels", "w_dim", "conv_clamp", "weight_gain"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        kernel_size: int = 1,
        conv_clamp: Optional[float] = None,
        channels_last: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).contiguous(memory_format=memory_format)
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x: Tensor, w: Tensor, fused_modconv: bool = True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act(x, self.bias, clamp=self.conv_clamp)
        return x

    def extra_repr(self):
        return f"in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}"


class SynthesisBlock(torch.nn.Module):
    __constants__ = [
        "in_channels",
        "w_dim",
        "resolution",
        "img_channels",
        "is_last",
        "architecture",
        "use_fp16",
        "channels_last",
        "fused_modconv_default",
        "num_conv",  # TODO these are only constant *after* initialization, is this how __constants__ actually works?
        "num_torgb",  # TODO
    ]

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
        fp16_channels_last: bool = False,  # Use channels-last memory format with FP16?
        fused_modconv_default: bool = True,  # Default value of fused_modconv.
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
        self.channels_last = use_fp16 and fp16_channels_last
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        self.const = torch.jit.annotate(Optional[torch.nn.Parameter], None)
        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        self.conv0 = torch.jit.annotate(Optional[SynthesisLayer], None)
        if in_channels != 0:
            self.conv0 = SynthesisLayer(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                up=2,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                **layer_kwargs,
            )
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            **layer_kwargs,
        )
        self.num_conv += 1

        if is_last or architecture == "skip":
            self.torgb = ToRGBLayer(
                out_channels, img_channels, w_dim=w_dim, conv_clamp=conv_clamp, channels_last=self.channels_last
            )
            self.num_torgb += 1

        self.skip = torch.jit.annotate(Optional[Conv2dLayer], None)
        if in_channels != 0 and architecture == "resnet":
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                up=2,
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(
        self,
        x: Optional[Tensor],
        img: Optional[Tensor],
        ws: Tensor,
        noise_mode: str = "const",
        fused_modconv: Optional[bool] = True,
        update_emas: bool = False,
        gain: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        _ = update_emas  # unused
        w_idx = 0
        memory_format = torch.channels_last if self.channels_last else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default

        # Input.
        if self.in_channels == 0 and self.const is not None:  # self.const is never None when reaching this case
            x = self.const.contiguous(memory_format=memory_format)
            x = torch.cat([x.unsqueeze(0) for _ in range(ws.shape[0])], dim=0)
        else:
            if x is None:
                x = torch.empty(0)  # never reached, need to let jit know x is not None
            x = x.contiguous(memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, ws[:, w_idx], noise_mode, fused_modconv, gain)
            w_idx += 1
        elif self.architecture == "resnet" and self.skip is not None:  # self.skip is never None when reaching this case
            y = self.skip(x, gain=sqrt(0.5))
            x = self.conv0(x, ws[:, w_idx], noise_mode, fused_modconv, gain)
            w_idx += 1
            x = self.conv1(x, ws[:, w_idx], noise_mode, fused_modconv, sqrt(0.5))
            w_idx += 1
            x = y.add_(x)
        elif self.conv0 is not None:  # self.conv0 is never None when reaching this case, but torch.jit needs some help
            x = self.conv0(x, ws[:, w_idx], noise_mode, fused_modconv, gain)
            w_idx += 1
            x = self.conv1(x, ws[:, w_idx], noise_mode, fused_modconv, gain)
            w_idx += 1

        # ToRGB.
        if img is not None:
            img = upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == "skip":
            y = self.torgb(x, ws[:, w_idx], fused_modconv)
            w_idx += 1
            y = y.contiguous(memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y
        if img is None:
            img = torch.empty(0)  # never reached

        return x, img

    def extra_repr(self):
        return f"resolution={self.resolution:d}, architecture={self.architecture:s}"


class SynthesisNetwork(torch.nn.Module):
    __constants__ = [
        "w_dim",
        "img_resolution",
        "img_resolution_log2",
        "img_channels",
        "num_fp16_res",
        "block_resolutions",
        "num_ws",
    ]

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

    def forward(
        self,
        ws: Tensor,
        c: Optional[Tensor] = None,
        noise_mode: str = "const",
        fused_modconv: Optional[bool] = None,
        update_emas: bool = False,
        gain: float = 1.0,
    ):
        block_ws = []
        w_idx = 0
        for block in self.bs:
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv

        x = img = torch.jit.annotate(Optional[Tensor], None)
        for b, block in enumerate(self.bs):
            x, img = block(x, img, block_ws[b], noise_mode, fused_modconv, update_emas, gain)
        return img

    def extra_repr(self):
        return " ".join(
            [
                f"w_dim={self.w_dim:d}, num_ws={self.num_ws:d},",
                f"img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},",
                f"num_fp16_res={self.num_fp16_res:d}",
            ]
        )


class Generator(torch.nn.Module):
    __constants__ = ["z_dim", "c_dim", "w_dim", "img_resolution", "img_channels", "num_ws"]

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
        update_emas: bool = False,
        fused_modconv: Optional[bool] = None,
        gain: float = 1.0,
    ):
        ws = self.mapping(
            z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas
        )
        img = self.synthesis(ws, update_emas=update_emas, c=c, fused_modconv=fused_modconv, gain=gain)
        return img
