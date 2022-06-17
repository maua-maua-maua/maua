"""Classifier-free guidance sampling from a diffusion model."""

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
from pathlib import Path

import clip
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange

from maua.utility import download

from .guided import parse_prompt

sys.path.append(os.path.dirname(__file__) + "/../../submodules/v_diffusion")

from diffusion import get_model, sampling, utils

URLS = {
    "cc12m_1_cfg": "https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1_cfg.pth",
}


def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])


def classifier_free(
    prompts,
    image_prompts=[],
    init=None,
    n=1,
    batch_size=1,
    side_x=256,
    side_y=256,
    steps=500,
    starting_timestep=0.9,
    model_name="cc12m_1_cfg",
    checkpoint=None,
    perceptor=None,
    eta=1.0,
    device=None,
    seed=None,
    outname="cfg_diffusion",
    save_intermediate=False,
):
    if seed is not None:
        torch.manual_seed(seed)

    if device is not None:
        device = torch.device(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name)()
    if checkpoint is None:
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

    zero_embed = torch.zeros([1, clip_model.visual.output_dim], device=device)
    target_embeds, weights = [zero_embed], []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in image_prompts:
        path, weight = parse_prompt(prompt)
        img = Image.open(utils.fetch(path)).convert("RGB")
        clip_size = clip_model.visual.input_resolution
        img = resize_and_center_crop(img, (clip_size, clip_size))
        batch = TF.to_tensor(img)[None].to(device)
        embed = F.normalize(clip_model.encode_image(normalize(batch)).float(), dim=-1)
        target_embeds.append(embed)
        weights.append(weight)

    weights = torch.tensor([1 - sum(weights), *weights], device=device)

    def cfg_model_fn(x, t):
        n = x.shape[0]
        n_conds = len(target_embeds)
        x_in = x.repeat([n_conds, 1, 1, 1])
        t_in = t.repeat([n_conds])
        clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
        vs = model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
        v = vs.mul(weights[:, None, None, None, None]).sum(0)
        return v

    def run(x, steps):
        return sampling.sample(cfg_model_fn, x, steps, eta, {})

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
        outs = run(x[i : i + cur_batch_size], steps)
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
    parser.add_argument("--size", type=str, default='256,256', help="the output image size: w,h")
    parser.add_argument("--steps", type=int, default=500, help="the number of timesteps")
    parser.add_argument("--starting_timestep", "-st", type=float, default=0.9, help="the timestep to start at (used with init images)")
    parser.add_argument("--eta", type=float, default=1.0, help="the amount of noise to add during sampling (0-1)")
    parser.add_argument("--model_name", type=str, default="cc12m_1_cfg", choices=["cc12m_1_cfg"], help="the model to use")
    parser.add_argument("--checkpoint", type=str, help="the checkpoint to use")
    parser.add_argument("--perceptor", type=str, default=None, choices=clip.available_models(), help="the CLIP perceptor to use (None defaults to version which diffusion model was trained with).")
    parser.add_argument("--device", type=str, help="the device to use")
    parser.add_argument("--seed", type=int, default=0, help="the random seed")
    parser.add_argument("--out_dir", type=str, default='output/', help="directory to save images to")
    parser.add_argument("--out_name", type=str, default=None, help="name to give images")
    # fmt: on
    return parser


def main(args):
    side_x, side_y = [int(v) for v in args.size.split(",")]
    out_name = (
        args.out_name
        if args.out_name is not None
        else "_".join([p.replace(" ", "_") for p in args.prompts] + [Path(i).stem for i in args.image_prompts])
        + "_cfg_diffusion"
    )
    imgs = classifier_free(
        prompts=args.prompts,
        image_prompts=args.image_prompts,
        init=args.init,
        n=args.n,
        batch_size=args.batch_size,
        side_x=side_x,
        side_y=side_y,
        steps=args.steps,
        starting_timestep=args.starting_timestep,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        perceptor=args.perceptor,
        eta=args.eta,
        device=args.device,
        seed=args.seed,
        outname=f"{args.out_dir}/{out_name}",
        save_intermediate=True,
    )
    for k, im in enumerate(imgs):
        im.save(f"{args.out_dir}/{out_name}_{k}.png")
