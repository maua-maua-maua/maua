import warnings
from typing import Tuple

import kornia.geometry.transform as kT
import numpy as np
import torch
from torch.nn.functional import interpolate, pad

from ..load import load_network
from ..studio.src.models import stylegan2
from . import MauaSynthesizer
from .stylegan import StyleGAN, StyleGANMapper


class StyleGAN2(StyleGAN):
    pass


class StyleGAN2Mapper(StyleGANMapper):
    MappingNetwork = stylegan2.MappingNetwork


class StyleGAN2Synthesizer(MauaSynthesizer):
    def __init__(self, model_file: str, output_size: Tuple[int, int], strategy: str, layer: int) -> None:
        super().__init__()

        if model_file is None or model_file == "None":
            self.G_synth = stylegan2.SynthesisNetwork(w_dim=512, img_resolution=1024, img_channels=3)
        else:
            self.G_synth: stylegan2.SynthesisNetwork = load_network(model_file).synthesis

        self.w_dim, self.num_ws = self.G_synth.w_dim, self.G_synth.num_ws
        self.layer_names = [
            f"b{block_size}.conv{1 if block_size == 4 else c % 2}"
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
        latent_w: torch.Tensor = None,
        latent_w_plus: torch.Tensor = None,
        translation: torch.Tensor = None,
        translation_layer: int = 7,
        zoom: torch.Tensor = None,
        zoom_layer: int = 7,
        zoom_center: int = None,
        rotation: torch.Tensor = None,
        rotation_layer: int = 7,
        rotation_center: int = None,
        **noise,
    ) -> torch.Tensor:
        if latent_w is None and latent_w_plus is None:
            raise Exception("One of latent_w or latent_w_plus inputs must be supplied!")
        if latent_w is not None and latent_w_plus is not None:
            warnings.warn("Both latent_w and latent_w_plus supplied, using latent_w_plus input...")

        if translation is not None:
            self.apply_translation(translation_layer, translation)
        if zoom is not None:
            self.apply_zoom(zoom_layer, zoom, zoom_center)
        if rotation is not None:
            self.apply_rotation(rotation_layer, rotation, rotation_center)

        if latent_w_plus is None:
            latent_w_plus = torch.tile(latent_w[:, None, :], (1, self.G_synth.num_ws, 1))

        if noise is not None:
            for l, (noise_l, layer_name) in enumerate(zip(noise.values(), self.layer_names[1:])):
                block, conv = layer_name.split(".")
                layer = getattr(getattr(self.G_synth, block), conv)

                noise_l = noise_l.to(layer.noise_const.device, non_blocking=True)

                if (noise_l.shape[-2], noise_l.shape[-1]) != (layer.noise_const.shape[-2], layer.noise_const.shape[-1]):
                    warnings.warn(
                        f"Supplied noise for layer {layer_name} ({l + 1}) has shape {noise_l.shape} while the expected "
                        f"shape is {layer.noise_const.shape}. Resizing the supplied noise to match..."
                    )
                    h, w = layer.noise_const.shape[-2], layer.noise_const.shape[-1]
                    noise_l = interpolate(noise_l, (h, w), mode="bicubic", align_corners=False)

                layer.noise_const.set_(noise_l)

        return self.G_synth.forward(latent_w_plus, noise_mode="const")

    def change_output_resolution(self, output_size: Tuple[int, int], strategy: str, layer: int):
        self.refresh_model_hooks()

        if output_size != (self.G_synth.img_resolution, self.G_synth.img_resolution):
            block, conv = self.layer_names[layer].split(".")

            layer_size = int(block.replace("b", ""))
            lay_mult = self.G_synth.img_resolution // layer_size
            unrounded_size = np.array(output_size) / lay_mult
            target_size = np.round(unrounded_size).astype(int)
            if sum(abs(unrounded_size - target_size)) > 1e-10:
                warnings.warn(
                    f"Layer {layer} resizes to multiples of {lay_mult}. --output-size rounded to {lay_mult * target_size}"
                )

            feat_hook, prev_img_hook, rgb_hook = get_hook(layer_size, target_size, strategy)

            synth_block = getattr(self.G_synth, block)
            synth_layer = getattr(synth_block, conv)
            torgb_layer = getattr(synth_block, "torgb")

            self._hook_handles.append(
                synth_layer.register_forward_pre_hook(feat_hook)
                if layer == 0
                else synth_layer.register_forward_hook(feat_hook)
            )
            if layer != 0:
                self._hook_handles.append(synth_block.register_forward_hook(prev_img_hook))
                self._hook_handles.append(torgb_layer.register_forward_hook(rgb_hook))

            for l in range(layer + 1, len(self.layer_names)):
                b, c = self.layer_names[l].split(".")
                noise_layer = getattr(getattr(self.G_synth, b), c)

                def noise_adjust(module, input, l=l):
                    if not hasattr(module, "noise_adjusted"):
                        x = input[0]
                        fac = 2 - l % 2
                        (_, _, h, w), dev, dtype = x.shape, x.device, x.dtype
                        del module.noise_const
                        setattr(module, "noise_const", torch.randn(1, 1, h * fac, w * fac, device=dev, dtype=dtype))
                        module.noise_adjusted = True

                self._hook_handles.append(noise_layer.register_forward_pre_hook(noise_adjust))

        self.output_size = output_size

    def apply_translation(self, layer, translation):
        block, conv = self.layer_names[layer].split(".")
        synth_layer = getattr(getattr(self.G_synth, block), conv)

        def translate_hook(module, input, output):
            _, _, h, w = output.shape
            output = kT.translate(
                output, translation * torch.tensor([[h, w]], device=translation.device), padding_mode="reflection"
            )
            return output

        if self.translate_hook:
            self.translate_hook.remove()
        self.translate_hook = synth_layer.register_forward_hook(translate_hook)

    def apply_rotation(self, layer, angle, center):
        block, conv = self.layer_names[layer].split(".")
        synth_layer = getattr(getattr(self.G_synth, block), conv)

        def rotation_hook(module, input, output):
            output = kT.rotate(output, angle.squeeze(), center, padding_mode="reflection")
            return output

        if self.rotate_hook:
            self.rotate_hook.remove()
        self.rotate_hook = synth_layer.register_forward_hook(rotation_hook)

    def apply_zoom(self, layer, zoom, center):
        block, conv = self.layer_names[layer].split(".")
        synth_layer = getattr(getattr(self.G_synth, block), conv)

        def zoom_hook(module, input, output):
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
            block, conv = layer.split(".")
            synth_layer = getattr(getattr(self.G_synth, block), conv)
            h, w = synth_layer.noise_const.shape[-2], synth_layer.noise_const.shape[-1]
            noises[f"noise{l}"] = interpolate(noise, (h, w), mode="bicubic", align_corners=False)
        return noises


def get_hook(layer_size, target_size, strategy, add_noise=True):
    target_size = np.flip(target_size)  # W,H --> H,W

    if strategy == "stretch":
        if add_noise:
            noise = torch.ones((1, 1, target_size[0], target_size[1])).cuda() * float("nan")

        def resize(x, feat=False):

            x = interpolate(x, tuple(target_size), mode="bicubic", align_corners=False)

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
                x += noise

            return x

        def inverse(x):
            return interpolate(x, (layer_size, layer_size), mode="bicubic", align_corners=False)

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
            value = 0  # not used

        if add_noise:
            noise = torch.ones((1, 1, target_size[0], target_size[1])).cuda() * float("nan")

        # TODO negative padding

        def resize(x, feat=False):
            x = pad(x, padding, mode=how, value=value)

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
                x += noise

            return x

        def inverse(x):
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

    def feat_hook(module, input, output=None):
        is_pre_hook = False
        if output is None:
            is_pre_hook = True
            output = input[0]

        output = resize(output, feat=True)

        if is_pre_hook:
            output = (output, *input[1:])
        return output

    def img_hook(module, input, output):
        return (output[0], resize(output[1], feat=False))

    def rgb_hook(module, input, output):
        return inverse(output)

    return feat_hook, img_hook, rgb_hook
