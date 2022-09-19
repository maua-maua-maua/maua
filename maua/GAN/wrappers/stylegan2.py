import os
import sys
import warnings
from typing import Dict, Optional, Tuple

import kornia.geometry.transform as kT
import numpy as np
import torch
from torch import Tensor

from ..load import load_network
from ..nv.networks import stylegan2 as stylegan2_train
from .inference import stylegan2 as stylegan2_inference
from .stylegan import StyleGAN, StyleGANMapper, StyleGANSynthesizer

SynthesisLayerInputType = Tuple[Tensor, Tensor, str, float]
SynthesisBlockInputType = Tuple[Optional[Tensor], Optional[Tensor], Tensor, str]
ToRGBInputType = Tuple[Tensor, Tensor]


class StyleGAN2Mapper(StyleGANMapper):
    MapperClsFn = lambda inference: (stylegan2_inference if inference else stylegan2_train).MappingNetwork


class StyleGAN2Synthesizer(StyleGANSynthesizer):
    __constants__ = ["w_dim", "num_ws", "layer_names"]

    def __init__(
        self, model_file: str, inference: bool, output_size: Optional[Tuple[int, int]], strategy: str, layer: int
    ) -> None:
        super().__init__()

        if model_file is None or model_file == "None":
            self.G_synth = (stylegan2_inference if inference else stylegan2_train).SynthesisNetwork(
                w_dim=512, img_resolution=1024, img_channels=3
            )
        else:
            self.G_synth: (stylegan2_inference if inference else stylegan2_train).SynthesisNetwork = load_network(
                model_file, inference
            ).synthesis
        if not hasattr(self.G_synth, "bs"):
            self.G_synth.bs = []
            for res in self.G_synth.block_resolutions:
                self.G_synth.bs.append(getattr(self.G_synth, f"b{res}"))

        if output_size is None:
            output_size = (self.G_synth.img_resolution, self.G_synth.img_resolution)

        self.w_dim, self.num_ws = self.G_synth.w_dim, self.G_synth.num_ws
        self.layer_names = [
            f"bs.{c//2}.conv{1 if block_size == 4 else c % 2}"
            for c, block_size in enumerate(sorted(self.G_synth.block_resolutions * 2))
        ]

        self.modulation_targets = {
            "latent_w": (self.w_dim,),
            "latent_w_plus": (self.num_ws, self.w_dim),
            "translation": (2,),
            "rotation": (1,),
        }

        self.translate_hook, self.rotate_hook, self.zoom_hook = None, None, None
        self.change_output_resolution(output_size, strategy, layer)

    def forward(
        self,
        latents: Tensor,
        translation: Optional[Tensor] = None,
        translation_layer: int = 7,
        zoom: Optional[Tensor] = None,
        zoom_layer: int = 7,
        zoom_center: Optional[int] = None,
        rotation: Optional[Tensor] = None,
        rotation_layer: int = 7,
        rotation_center: Optional[int] = None,
        **noise,
    ) -> Tensor:
        if translation is not None:
            self.apply_translation(translation_layer, translation)
        if zoom is not None:
            self.apply_zoom(zoom_layer, zoom, zoom_center)
        if rotation is not None:
            self.apply_rotation(rotation_layer, rotation, rotation_center)

        if noise is not None:
            noises, l = list(noise.values()), 0
            for block in self.G_synth.bs:
                if l >= len(noises):
                    continue
                for c in ([block.conv0] if hasattr(block, "conv0") else []) + [block.conv1]:
                    noise_l = noises[l].to(c.noise_const, non_blocking=True)
                    if (noise_l.shape[-2], noise_l.shape[-1]) != (c.noise_const.shape[-2], c.noise_const.shape[-1]):
                        warnings.warn(
                            f"Supplied noise for SynthesisLayer {l} has shape {noise_l.shape} while the expected "
                            f"shape is {c.noise_const.shape}. Resizing the supplied noise to match..."
                        )
                        h, w = c.noise_const.shape[-2], c.noise_const.shape[-1]
                        noise_l = torch.nn.functional.interpolate(noise_l, (h, w), mode="bicubic", align_corners=False)
                    setattr(c, "noise_const", noise_l)
                    l += 1

        return self.G_synth.forward(latents, noise_mode="const")

    def change_output_resolution(self, output_size: Tuple[int, int], strategy: str, layer: int):
        self.refresh_model_hooks()

        if output_size != (self.G_synth.img_resolution, self.G_synth.img_resolution):
            _, block, conv = self.layer_names[layer].split(".")

            synth_block = self.G_synth.bs[int(block)]
            synth_layer = getattr(synth_block, conv)
            torgb_layer = getattr(synth_block, "torgb")

            layer_size = synth_layer.resolution
            lay_mult = self.G_synth.img_resolution // layer_size
            unrounded_size = np.array(output_size) / lay_mult
            target_size = np.round(unrounded_size).astype(int)
            if sum(abs(unrounded_size - target_size)) > 1e-10:
                warnings.warn(
                    f"Layer {layer} resizes to multiples of {lay_mult}. --output-size rounded to {lay_mult * target_size}"
                )

            use_pre_hook = layer == 0
            feat_hook, prev_img_hook, rgb_hook = get_hook(layer_size, target_size, strategy, pre=use_pre_hook)

            self._hook_handles.append(
                synth_layer.register_forward_pre_hook(feat_hook)
                if layer == 0
                else synth_layer.register_forward_hook(feat_hook)
            )
            if not use_pre_hook:
                self._hook_handles.append(synth_block.register_forward_hook(prev_img_hook))
                self._hook_handles.append(torgb_layer.register_forward_hook(rgb_hook))

            for l in range(layer + 1, len(self.layer_names)):
                _, b, c = self.layer_names[l].split(".")
                noise_layer = getattr(self.G_synth.bs[int(b)], c)

                def noise_adjust(mod, input: Tuple[Tensor, Tensor, str, bool, float]) -> None:
                    if not hasattr(mod, "noise_adjusted") or not mod.noise_adjusted:
                        _, _, h, w = input[0].shape
                        dev, dtype = mod.noise_const.device, mod.noise_const.dtype
                        mod.noise_const = torch.randn((1, 1, h * mod.up, w * mod.up), device=dev, dtype=dtype)
                        # mod.noise_strength = torch.nn.Parameter(torch.ones((1)).to(input[0]))
                        mod.noise_adjusted = True

                self._hook_handles.append(noise_layer.register_forward_pre_hook(noise_adjust))

            self.forward(torch.randn((1, 18, 512)))  # ensure that noise_adjust hooks have adjusted noise buffers

        self.output_size = output_size

    def apply_translation(self, layer, translation):
        _, block, conv = self.layer_names[layer].split(".")
        synth_layer = getattr(self.G_synth.bs[int(block)], conv)

        def translate_hook(
            module, input: Tuple[Tensor, Tensor, str, bool, float], output: Tuple[Tensor]
        ) -> Tuple[Tensor]:
            _, _, h, w = output.shape
            output = kT.translate(
                output, translation * torch.tensor([[h, w]], device=translation.device), padding_mode="reflection"
            )
            return output

        if self.translate_hook:
            self.translate_hook.remove()
        self.translate_hook = synth_layer.register_forward_hook(translate_hook)

    def apply_rotation(self, layer, angle, center):
        _, block, conv = self.layer_names[layer].split(".")
        synth_layer = getattr(self.G_synth.bs[int(block)], conv)

        def rotation_hook(
            module, input: Tuple[Tensor, Tensor, str, bool, float], output: Tuple[Tensor]
        ) -> Tuple[Tensor]:
            output = kT.rotate(output, angle.squeeze(), center, padding_mode="reflection")
            return output

        if self.rotate_hook:
            self.rotate_hook.remove()
        self.rotate_hook = synth_layer.register_forward_hook(rotation_hook)

    def apply_zoom(self, layer, zoom, center):
        _, block, conv = self.layer_names[layer].split(".")
        synth_layer = getattr(self.G_synth.bs[int(block)], conv)

        def zoom_hook(module, input: Tuple[Tensor, Tensor, str, bool, float], output: Tuple[Tensor]) -> Tuple[Tensor]:
            output = kT.scale(output, zoom.squeeze(), center, padding_mode="reflection")
            return output

        if self.zoom_hook:
            self.zoom_hook.remove()
        self.zoom_hook = synth_layer.register_forward_hook(zoom_hook)

    def make_noise_pyramid(self, noise, layer_limit=8):
        noises = {}
        for l, layer in enumerate(self.layer_names[1:]):
            if l > layer_limit:
                continue
            _, block, conv = layer.split(".")
            synth_layer = getattr(self.G_synth.bs[int(block)], conv)
            h, w = synth_layer.noise_const.shape[-2], synth_layer.noise_const.shape[-1]
            try:
                noises[f"noise{l}"] = torch.nn.functional.interpolate(
                    noise, (h, w), mode="bicubic", align_corners=False
                ).cpu()
            except RuntimeError:  # CUDA out of memory
                noises[f"noise{l}"] = torch.nn.functional.interpolate(
                    noise.cpu(), (h, w), mode="bicubic", align_corners=False
                )
            noises[f"noise{l}"] /= noises[f"noise{l}"].std((1, 2, 3), keepdim=True)
        return noises


def get_hook(layer_size, target_size, strategy, add_noise=True, pre=False):
    target_size = np.flip(target_size)  # W,H --> H,W

    if strategy == "stretch":
        noise = torch.ones((1, 1, target_size[0], target_size[1])).cuda() * float("nan") if add_noise else None

        def resize(
            x: Tensor,
            feat: bool = False,
            target_size: Tuple[int, int] = tuple(target_size),
            noise: Tensor = noise,
            add_noise: bool = add_noise,
        ):

            x = torch.nn.functional.interpolate(x, target_size, mode="bicubic", align_corners=False)

            if feat and add_noise:
                if noise.isnan().any():
                    h, w = target_size
                    channel_noises = [
                        torch.normal(
                            mean=x[:, c].mean().item(),
                            std=x[:, c].std().item(),
                            size=(1, 1, h, w),
                            device=noise.device,
                            dtype=noise.dtype,
                        )
                        for c in range(x.size(1))
                    ]
                    noise.set_(torch.cat(channel_noises, dim=1))
                x += noise

            return x

        def inverse(x: Tensor, layer_size: int = layer_size):
            return torch.nn.functional.interpolate(x, (layer_size, layer_size), mode="bicubic", align_corners=False)

    elif strategy.startswith("pad"):
        _, how, where = strategy.split("-")

        pad_h, pad_w = (target_size - layer_size).round().astype(int)

        if where == "out":
            padding = (pad_w // 2, round(1e-16 + pad_w / 2), pad_h // 2, round(1e-16 + pad_h / 2))
        if where == "left":
            padding = (pad_w, 0, pad_h // 2, round(1e-16 + pad_h / 2))
        if where == "right":
            padding = (0, pad_w, pad_h // 2, round(1e-16 + pad_h / 2))
        if where == "top":
            padding = (pad_w // 2, round(1e-16 + pad_w / 2), pad_h, 0)
        if where == "bottom":
            padding = (pad_w // 2, round(1e-16 + pad_w / 2), 0, pad_h)

        if how not in ["reflect", "replicate", "circular"]:
            value = float(how)
            how = "constant"
        else:
            value = 0.0  # not used

        if add_noise:
            noise = torch.ones((1, 1, target_size[0], target_size[1])).cuda() * float("nan")

        # TODO negative padding

        def resize(
            x: Tensor,
            feat: bool = False,
            padding: Tuple[int, int, int, int] = padding,
            how: str = how,
            value: float = value,
            target_size: Tuple[int, int] = tuple(target_size),
            noise: Tensor = noise,
        ):
            x = torch.nn.functional.pad(x, padding, mode=how, value=value)

            if feat:
                if noise.isnan().any():
                    h, w = target_size
                    channel_noises = [
                        torch.normal(
                            mean=x[:, c].mean().item(),
                            std=x[:, c].std().item(),
                            size=(1, 1, h, w),
                            device=noise.device,
                            dtype=noise.dtype,
                        )
                        for c in range(x.size(1))
                    ]
                    noise.set_(torch.cat(channel_noises, dim=1))
                x += noise.to(x)

            return x

        def inverse(x: Tensor, padding: Tuple[int, int, int, int] = padding):
            if padding[3] == 0:
                if padding[1] == 0:
                    return x[..., padding[2] :, padding[0] :]
                else:
                    return x[..., padding[2] :, padding[0] : -padding[1]]
            else:
                if padding[1] == 0:
                    return x[..., padding[2] : -padding[3], padding[0] :]
                else:
                    return x[..., padding[2] : -padding[3], padding[0] : -padding[1]]

    else:
        raise Exception(f"Resize strategy not found: {strategy}")

    if pre:

        def feat_hook(module, input: SynthesisLayerInputType) -> SynthesisLayerInputType:
            return (resize(input[0], feat=True), *input[1:])

    else:

        def feat_hook(module, input: SynthesisLayerInputType, output: Tensor) -> Tensor:
            return resize(output, feat=True)

    def img_hook(module, input: SynthesisBlockInputType, output: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return (output[0], resize(output[1], feat=False))

    def rgb_hook(module, input: ToRGBInputType, output: Tensor) -> Tensor:
        return inverse(output)

    return feat_hook, img_hook, rgb_hook


class StyleGAN2(StyleGAN):
    SynthesizerCls = StyleGAN2Synthesizer
    MapperCls = StyleGAN2Mapper
