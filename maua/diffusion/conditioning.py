from functools import partial
from pathlib import Path

import clip
import lpips
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import l1_loss
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_pil_image, to_tensor

from ..ops.image import resample
from ..ops.loss import range_loss, spherical_dist_loss, tv_loss
from ..utility import fetch
from .cutouts import make_cutouts


class TextPrompt(torch.nn.Module):
    def __init__(self, text, weight=1.0):
        super().__init__()
        self.text = text
        self.weight = weight

    def forward(self):
        return self.text, self.weight


class ImagePrompt(torch.nn.Module):
    def __init__(self, img=None, path=None, size=None, weight=1.0):
        super().__init__()
        self.weight = weight

        if path is not None:
            allowed_types = (str, Path)
            assert isinstance(path, allowed_types), f"path must be one of {allowed_types}"
            img = Image.open(fetch(path)).convert("RGB")
            self.img = to_tensor(img).unsqueeze(0)

        elif img is not None:
            allowed_types = (Image.Image, torch.Tensor, np.ndarray)
            assert isinstance(img, allowed_types), f"img must be one of {allowed_types}"
            if isinstance(img, (Image.Image, np.ndarray)):
                self.img = to_tensor(img).unsqueeze(0)
            else:
                self.img = img
                assert self.img.dim() == 4, "img must be of shape (B, C, H, W)"

        else:
            raise Exception("path or img must be specified")

        if size is not None:
            self.img = resample(self.img, min(size))

    def forward(self):
        return self.img, self.weight


class StylePrompt(ImagePrompt):
    pass


class ContentPrompt(ImagePrompt):
    pass


class GradModule(torch.nn.Module):
    def set_targets(self, prompts):
        pass

    def forward(self, img):
        return 0


class CLIPGrads(GradModule):
    def __init__(
        self,
        scale=1,
        perceptors=["ViT-B/32", "ViT-B/16", "RN50"],
        cutouts="maua",
        cutout_kwargs=dict(cutn=16),
        cutout_batches=4,
        clamp_gradient=0.15,
    ):
        super().__init__()
        self.scale = scale
        self.clip_models = torch.nn.ModuleList(
            [clip.load(name, jit=False)[0].eval().requires_grad_(False) for name in perceptors]
        )
        self.normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.cutouts = torch.nn.ModuleList(
            [
                make_cutouts(cutouts, cut_size=clip_model.visual.input_resolution, **cutout_kwargs)
                for clip_model in self.clip_models
            ]
        )
        self.cutout_batches = cutout_batches
        self.clamp_gradient = clamp_gradient

    def set_targets(self, prompts):
        target_embeds, weights = [[] for _ in range(len(self.clip_models))], []
        device = next(self.clip_models[0].parameters()).device

        for prompt in prompts:

            if isinstance(prompt, TextPrompt):
                txt, weight = prompt()
                tokens = clip.tokenize(txt, truncate=True).to(device)
                weights.append(weight)
                for c, clip_model in enumerate(self.clip_models):
                    target_embeds[c].append(clip_model.encode_text(tokens).float())

            elif isinstance(prompt, StylePrompt):
                img, weight = prompt()
                img = img.to(device)
                for c, clip_model in enumerate(self.clip_models):
                    im_cuts = clip_model.encode_image(self.normalize(self.cutouts[c](img, t=0))).float()
                    target_embeds[c].append(im_cuts)
                    weights.extend([0.5 * weight / im_cuts.shape[0]])

        for c, target_embed in enumerate(target_embeds):
            self.clip_models[c].register_buffer("target", torch.cat(target_embed).unsqueeze(0))
            # register buffers on each clip_model because nn.BufferList doesn't exist https://github.com/pytorch/pytorch/issues/35735

        weights = torch.tensor(weights, device=device, dtype=torch.float)
        if weights.sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")
        weights /= weights.sum().abs()
        self.register_buffer("weights", weights)

    def forward(self, img, t):
        grad = torch.zeros_like(img)
        for c, clip_model in enumerate(self.clip_models):
            for _ in range(self.cutout_batches):
                image_embeds = clip_model.encode_image(self.normalize(self.cutouts[c](img.add(1).div(2), t))).float()
                dists = spherical_dist_loss(image_embeds.unsqueeze(1), clip_model.target)
                loss = dists.view((-1, img.shape[0], dists.shape[-1])).mul(self.weights).sum(2).mean(0)
                grad += torch.autograd.grad(loss.sum() * self.scale, img)[0] / self.cutout_batches

        if self.clamp_gradient:
            magnitude = grad.square().mean().sqrt()
            grad *= magnitude.clamp(max=self.clamp_gradient) / magnitude
        return grad


class LossGrads(GradModule):
    def __init__(self, loss_fn, scale=1):
        super().__init__()
        self.loss_fn = loss_fn
        self.scale = scale

    def forward(self, img, t):
        loss = self.loss_fn(img).sum() * self.scale
        grad = torch.autograd.grad(loss, img)[0]
        return grad


class LPIPSGrads(GradModule):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale
        self.lpips_model = lpips.LPIPS(net="vgg", verbose=False)

    def set_targets(self, prompts):
        for prompt in prompts:
            if isinstance(prompt, ContentPrompt):
                img, _ = prompt()
                self.register_buffer("target", img)

    def forward(self, img, t):
        if hasattr(self, "target"):
            loss = self.lpips_model(resample(img, 256), resample(self.target, 256)).sum() * self.scale
            grad = torch.autograd.grad(loss, img)[0]
        else:
            grad = torch.zeros_like(img)
        return grad


class GradientGuidedConditioning(torch.nn.Module):
    def __init__(self, diffusion, model, grad_modules, fast=True):
        super().__init__()
        self.fast = fast
        if self.fast:
            self.model = model
        else:
            self.model = partial(diffusion.p_mean_variance, model=model, clip_denoised=False)

        self.grad_modules = torch.nn.ModuleList(grad_modules)
        self.timestep_map = diffusion.timestep_map

        sqrt_alphas_cumprod = torch.from_numpy(diffusion.sqrt_alphas_cumprod).float()
        sqrt_one_minus_alphas_cumprod = torch.from_numpy(diffusion.sqrt_one_minus_alphas_cumprod).float()
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

    def set_targets(self, prompts):
        for grad_module in self.grad_modules:
            grad_module.set_targets(prompts)

    def forward(self, x, t, y=None):
        ot = t.clone()
        t = torch.tensor([self.timestep_map.index(t) for t in t.long()], device=x.device, dtype=torch.long)

        with torch.enable_grad():
            x = x.detach().requires_grad_()

            if self.fast:
                alpha = self.sqrt_alphas_cumprod[t]
                sigma = self.sqrt_one_minus_alphas_cumprod[t]
                cosine_t = torch.atan2(sigma, alpha) * 2 / torch.pi
                out = self.model(x, cosine_t).pred
                img = out * sigma + x * (1 - sigma)
            else:
                out = self.model(x=x, t=t, model_kwargs={"y": y})
                fac = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
                img = out["pred_xstart"] * fac + x * (1 - fac)

            img_grad = torch.zeros_like(img)
            for grad_mod in self.grad_modules:
                img_grad += grad_mod(img, ot)

            grad = -torch.autograd.grad(img, x, img_grad)[0]

        return grad
