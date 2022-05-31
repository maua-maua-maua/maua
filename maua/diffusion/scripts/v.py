"""CLIP guided sampling from a diffusion model."""

# Copyright (c) 2021 Katherine Crowson and John David Pressman

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import clip
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange

from maua.ops.image import random_cutouts
from maua.ops.loss import spherical_dist_loss
from maua.utility import download

from .guided import parse_prompt

sys.path.append(os.path.dirname(__file__) + "/../../submodules/v_diffusion")

from diffusion import get_model, sampling, utils

MODULE_DIR = Path(__file__).resolve().parent
URLS = {
    "cc12m_1": "https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1.pth",
    "yfcc_1": "https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_1.pth",
    "yfcc_2": "https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_2.pth",
}


def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])


def v_diffusion(
    prompts,
    image_prompts=[],
    init=None,
    n=1,
    batch_size=1,
    side_x=512,
    side_y=512,
    guidance_scale=2000,
    cutn=32,
    cut_pow=1.0,
    eta=1.0,
    model_name="yfcc_2",
    checkpoint=None,
    perceptor=None,
    device=None,
    starting_timestep=0.9,
    steps=1000,
    seed=None,
    outname="output/v_diffusion",
    save_intermediate=False,
):
    if seed is not None:
        torch.manual_seed(seed)

    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name)()
    if not checkpoint:
        checkpoint = f"modelzoo/v_diffusion_{model_name}.pth"
    if not os.path.exists(checkpoint):
        download(URLS[model_name], checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    if device.type == "cuda":
        model = model.half()
    model = model.to(device).eval().requires_grad_(False)
    clip_model_name = (
        perceptor if perceptor is not None else (model.clip_model if hasattr(model, "clip_model") else "ViT-B/16")
    )
    clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    if init:
        init = Image.open(utils.fetch(init)).convert("RGB")
        init = resize_and_center_crop(init, (side_x, side_y))
        init = utils.from_pil_image(init).cuda()[None].repeat([n, 1, 1, 1])

    target_embeds, weights = [], []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in image_prompts:
        path, weight = parse_prompt(prompt)
        img = Image.open(utils.fetch(path)).convert("RGB")
        img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        batch = random_cutouts(TF.to_tensor(img)[None].to(device), clip_model.visual.input_resolution, cutn, cut_pow)
        embeds = F.normalize(clip_model.encode_image(normalize(batch)).float(), dim=-1)
        target_embeds.append(embeds)
        weights.extend([weight / cutn] * cutn)

    if not target_embeds:
        raise RuntimeError("At least one text or image prompt must be specified.")
    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError("The weights must not sum to 0.")
    weights /= weights.sum().abs()

    clip_embed = F.normalize(target_embeds.mul(weights[:, None]).sum(0, keepdim=True), dim=-1)
    clip_embed = clip_embed.repeat([n, 1])

    def cond_fn(x, t, pred, clip_embed):
        clip_in = normalize(random_cutouts((pred + 1) / 2, clip_model.visual.input_resolution, cutn, cut_pow))
        image_embeds = clip_model.encode_image(clip_in).view([cutn, x.shape[0], -1])
        losses = spherical_dist_loss(image_embeds, clip_embed[None])
        loss = losses.mean(0).sum() * guidance_scale
        grad = -torch.autograd.grad(loss, x)[0]
        return grad

    def run(x, steps, clip_embed):
        if hasattr(model, "clip_model"):
            extra_args = {"clip_embed": clip_embed}
            cond_fn_ = cond_fn
        else:
            extra_args = {}
            cond_fn_ = partial(cond_fn, clip_embed=clip_embed)
        if not guidance_scale:
            return sampling.sample(model, x, steps, eta, extra_args)
        return sampling.cond_sample(model, x, steps, eta, extra_args, cond_fn_)

    x = torch.randn([n, 3, side_y, side_x], device=device)
    t = torch.linspace(1, 0, steps + 1, device=device)[:-1]
    steps = utils.get_spliced_ddpm_cosine_schedule(t)

    if init:
        steps = steps[steps < starting_timestep]
        alpha, sigma = utils.t_to_alpha_sigma(steps[0])
        x = init * alpha + x * sigma

    results = []
    for i in trange(0, n, batch_size):
        cur_batch_size = min(n - i, batch_size)
        outs = run(x[i : i + cur_batch_size], steps, clip_embed[i : i + cur_batch_size])
        for j, out in enumerate(outs):
            results.append(utils.to_pil_image(out))
            if save_intermediate:
                results[-1].save(f"{outname}_{i+j}.png")

    return results


def argument_parser():
    # fmt: off
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prompts", type=str, default=[], nargs="*", help="the text prompts to use")
    parser.add_argument("--image_prompts", type=str, default=[], nargs="*", metavar="IMAGE", help="the image prompts")
    parser.add_argument("--init", type=str, help="the init image")
    parser.add_argument("-n", type=int, default=1, help="the number of images to sample")
    parser.add_argument("--batch_size", "-bs", type=int, default=1, help="the number of images per batch")
    parser.add_argument("--size", type=str, default='512,512', help="the output image size: w,h")
    parser.add_argument("--guidance_scale", "-cs", type=float, default=2000.0, help="the CLIP guidance scale")
    parser.add_argument("--cutn", type=int, default=16, help="the number of random crops to use")
    parser.add_argument("--cut_pow", type=float, default=1.0, help="the random crop size power")
    parser.add_argument("--eta", type=float, default=1.0, help="the amount of noise to add during sampling (0-1)")
    parser.add_argument("--model_name", type=str, default="yfcc_2", choices=URLS.keys(), help="the model to use")
    parser.add_argument("--checkpoint", type=str, help="the checkpoint to use")
    parser.add_argument("--perceptor", type=str, default=None, choices=clip.available_models(), help="the CLIP perceptor to use (None defaults to version which diffusion model was trained with).")
    parser.add_argument("--device", type=str, help="the device to use")
    parser.add_argument("--starting_timestep", "-st", type=float, default=0.9, help="the timestep to start at (used with init images)")
    parser.add_argument("--steps", type=int, default=1000, help="the number of timesteps")
    parser.add_argument("--seed", type=int, default=None, help="the random seed")
    parser.add_argument("--out_dir", type=str, default='output/', help="directory to save images to")
    parser.add_argument("--out_name", type=str, default=None, help="name to give images")
    # fmt: on
    return parser


def main(args):
    side_x, side_y = [int(side) for side in args.size.split(",")]
    out_name = (
        args.out_name
        if args.out_name is not None
        else "_".join([p.replace(" ", "_") for p in args.prompts] + [Path(i).stem for i in args.image_prompts])
        + "_v_diffusion"
    )
    imgs = v_diffusion(
        prompts=args.prompts,
        image_prompts=args.image_prompts,
        init=args.init,
        n=args.n,
        batch_size=args.batch_size,
        side_x=side_x,
        side_y=side_y,
        guidance_scale=args.guidance_scale,
        cutn=args.cutn,
        cut_pow=args.cut_pow,
        eta=args.eta,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        perceptor=args.perceptor,
        device=args.device,
        starting_timestep=args.starting_timestep,
        steps=args.steps,
        seed=args.seed,
        outname=f"{args.out_dir}/{out_name}",
        save_intermediate=True,
    )
    for k, im in enumerate(imgs):
        im.save(f"{args.out_dir}/{out_name}_{k}.png")
