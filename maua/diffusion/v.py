import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange
from utility import download

from .util import MakeCutouts, parse_prompt, spherical_dist_loss

sys.path += ["submodules/", "submodules/v_diffusion"]

from maua.submodules.CLIP import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from diffusion import get_model, sampling, utils

MODULE_DIR = Path(__file__).resolve().parent
URLS = {
    "cc12m_1": "https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1.pth",
    "yfcc_1": "https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_1.pth",
    "yfcc_2": "https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_2.pth",
}


def v_diffusion(
    prompts: List[str] = [],  # the text prompts to use
    images: List[str] = [],  # the image prompts
    batch_size: int = 1,  # the number of images per batch
    image_size: str = "512,512",  # The width,height of output image in pixels
    checkpoint: Optional[str] = None,  # the checkpoint to use
    out_dir: str = "output/",  # the path to save images
    guidance_scale: float = 500.0,  # the CLIP guidance scale
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # device to run on
    eta: float = 1.0,  # the amount of noise to add during sampling (0-1)
    model: str = "yfcc_2",  # the model to use [cc12m_1, yfcc_1, yfcc_2]
    n: int = 1,  # the number of images to sample
    seed: int = 0,  # the random seed
    steps: int = 1000,  # the number of timesteps
):
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = out_dir

    model = get_model(model)()
    checkpoint = checkpoint
    if not checkpoint:
        checkpoint = MODULE_DIR / f"modelzoo/v_diffusion_{model}.pth"
    if not os.path.exists(checkpoint):
        download(URLS[model], str(checkpoint))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    if device.type == "cuda":
        model = model.half()
    model = model.to(device).eval().requires_grad_(False)
    clip_model = clip.load(model.clip_model, jit=False, device=device)[0]
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    side_x, side_y = [int(side) for side in image_size.split(",")]
    cutn = 128
    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, cutn=cutn, cut_pow=1)

    target_embeds, weights = [], []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in images:
        path, weight = parse_prompt(prompt)
        img = Image.open(utils.fetch(path)).convert("RGB")
        img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        batch = make_cutouts(TF.to_tensor(img)[None].to(device))
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

    torch.manual_seed(seed)

    def cond_fn(x, t, pred, clip_embed):
        clip_in = normalize(make_cutouts((pred + 1) / 2))
        image_embeds = clip_model.encode_image(clip_in).view([cutn, x.shape[0], -1])
        losses = spherical_dist_loss(image_embeds, clip_embed[None])
        loss = losses.mean(0).sum() * guidance_scale
        grad = -torch.autograd.grad(loss, x)[0]
        return grad

    def run(x, clip_embed):
        t = torch.linspace(1, 0, steps + 1, device=device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        extra_args = {"clip_embed": clip_embed}
        if not guidance_scale:
            return sampling.sample(model, x, steps, eta, extra_args)
        return sampling.cond_sample(model, x, steps, eta, extra_args, cond_fn)

    def run_all(n, batch_size):
        x = torch.randn([n, 3, side_y, side_x], device=device)
        for i in trange(0, n, batch_size):
            cur_batch_size = min(n - i, batch_size)
            outs = run(x[i : i + cur_batch_size], clip_embed[i : i + cur_batch_size])
            for j, out in enumerate(outs):
                yield utils.to_pil_image(out)

    run_all(n, batch_size)
