import argparse
import os
import sys
from pathlib import Path

import clip
import lpips
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from maua.dalle.ru.finetune import argument_parser
from maua.ops.image import random_cutouts, resample
from maua.ops.loss import range_loss, spherical_dist_loss, tv_loss
from maua.utility import download, fetch

sys.path += ["maua/submodules/guided_diffusion"]

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


def parse_prompt(prompt):
    if prompt.startswith("http://") or prompt.startswith("https://"):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])


def create_models(
    perceptor_names=["ViT-B/32"],
    image_size=512,
    timestep_respacing="1000",
    diffusion_steps=1000,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    model_config = model_and_diffusion_defaults()
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": diffusion_steps,
            "rescale_timesteps": True,
            "timestep_respacing": timestep_respacing,
            "image_size": image_size,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_fp16": True,
            "use_scale_shift_norm": True,
            # "use_checkpoint": True,
        }
    )
    diffusion_model, diffusion = create_model_and_diffusion(**model_config)
    if image_size == 512:
        checkpoint = "modelzoo/512x512_diffusion_uncond_finetune_008100.pt"
        if not os.path.exists(checkpoint):
            download(
                "https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt",
                checkpoint,
            )
    elif image_size == 256:
        checkpoint = "modelzoo/256x256_diffusion_uncond.pt"
        if not os.path.exists(checkpoint):
            download(
                "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
                checkpoint,
            )
    diffusion_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    diffusion_model.requires_grad_(False).eval().to(device)
    for name, param in diffusion_model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        diffusion_model.convert_to_fp16()

    if model_config["timestep_respacing"].startswith("ddim"):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    clip_models = [clip.load(name, jit=False)[0].eval().requires_grad_(False).to(device) for name in perceptor_names]
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(device)

    return diffusion_model, diffusion, sample_fn, clip_models, normalize, lpips_model


@torch.no_grad()
def initialize(prompts, image_prompts, init_image, side_x, side_y, cutn, cut_pow, clip_models, device):
    target_embeds, weights = [[] for _ in range(len(clip_models))], []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        for c, clip_model in enumerate(clip_models):
            target_embeds[c].append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in image_prompts:
        if isinstance(prompt, torch.Tensor):
            img = resample(prompt, min(side_x, side_y))
        else:
            path, weight = parse_prompt(prompt)
            img = Image.open(fetch(path)).convert("RGB")
            img = TF.to_tensor(
                TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
            ).unsqueeze(0)
        img = img.to(device)
        for c, clip_model in enumerate(clip_models):
            target_embeds[c].append(
                clip_model.encode_image(
                    normalize(random_cutouts(img, clip_model.visual.input_resolution, cutn, cut_pow))
                ).float()
            )
        weights.extend([0.5 * weight / cutn] * cutn)

    target_embeds = [torch.cat(target_embed) for target_embed in target_embeds]
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError("The weights must not sum to 0.")
    weights /= weights.sum().abs()

    init = None
    if init_image is not None:
        if isinstance(init_image, torch.Tensor):
            init = resample(init_image, (side_y, side_x))
        else:
            init = Image.open(fetch(init_image)).convert("RGB")
            init = TF.to_tensor(init.resize((side_x, side_y), Image.LANCZOS)).unsqueeze(0)
        init = init.to(device).mul(2).sub(1)

    return target_embeds, weights, init


diffusion_model = None


def guided_diffusion(
    prompts,
    image_prompts=[],
    init_image=None,
    side_x=512,
    side_y=512,
    diffusion_model_size=512,
    batch_size=1,
    clip_guidance_scale=5000,
    tv_scale=100,
    range_scale=50,
    init_scale=0,
    diffusion_steps=1000,
    timestep_respacing="1000",
    skip_timesteps=0,
    cutn=64,
    cutn_batches=1,
    cut_pow=0.5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    perceptor_names=["ViT-B/32"],
    seed=None,
    outname="output/guided_diffusion",
    save_intermediate=False,
    cache_model=True,
):
    if seed is not None:
        torch.manual_seed(seed)

    if cache_model:
        global diffusion_model, diffusion, sample_fn, clip_models, normalize, lpips_model
        if diffusion_model is None:
            diffusion_model, diffusion, sample_fn, clip_models, normalize, lpips_model = create_models(
                perceptor_names, diffusion_model_size, timestep_respacing, diffusion_steps, device
            )
    else:
        diffusion_model, diffusion, sample_fn, clip_models, normalize, lpips_model = create_models(
            perceptor_names, diffusion_model_size, timestep_respacing, diffusion_steps, device
        )
    sqrt_one_minus_alphas_cumprod = torch.from_numpy(diffusion.sqrt_one_minus_alphas_cumprod).to(device)

    target_embeds, weights, init = initialize(
        prompts, image_prompts, init_image, side_x, side_y, cutn, cut_pow, clip_models, device
    )

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            cur_t = torch.round(t * int(timestep_respacing) / diffusion_steps).long()
            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            out = diffusion.p_mean_variance(diffusion_model, x, my_t, clip_denoised=False, model_kwargs={"y": y})
            fac = sqrt_one_minus_alphas_cumprod[cur_t].reshape(-1, 1, 1, 1)
            x_in = out["pred_xstart"] * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
            clip_in = x_in.add(1).div(2)
            losses = []
            for clip_model, target_embed in zip(clip_models, target_embeds):
                for i in range(cutn_batches):
                    image_embeds = clip_model.encode_image(
                        normalize(random_cutouts(clip_in, clip_model.visual.input_resolution, cutn, cut_pow))
                    ).float()
                    dists = spherical_dist_loss(image_embeds.unsqueeze(1), target_embed.unsqueeze(0))
                    dists = dists.view([cutn, n, -1])
                    losses.append(dists.mul(weights).sum(2).mean(0))
            x_in_grad += torch.autograd.grad(torch.cat(losses).sum() * clip_guidance_scale, x_in)[0] / cutn_batches
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out["pred_xstart"])
            loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale
            if init is not None and init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            return grad

    for j, sample in enumerate(
        sample_fn(
            diffusion_model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init,
            randomize_class=True,
        )
    ):
        if save_intermediate and j % 25 == 0:
            for k, image in enumerate(sample["pred_xstart"]):
                TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(f"{outname}_{seed}_{k}.png")
    return sample["pred_xstart"].add(1).div(2).clamp(0, 1)


def argument_parser():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--image_prompts", nargs="*", default=[])
    parser.add_argument("--init_image", default=None)
    parser.add_argument("--size", default="512,512")
    parser.add_argument("--diffusion_model_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--clip_guidance_scale", type=float, default=5000)
    parser.add_argument("--tv_scale", type=float, default=100)
    parser.add_argument("--range_scale", type=float, default=50)
    parser.add_argument("--init_scale", type=float, default=0)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--timestep_respacing", type=str, default="1000")
    parser.add_argument("--skip_timesteps", type=int, default=0)
    parser.add_argument("--cutn", type=int, default=64)
    parser.add_argument("--cutn_batches", type=int, default=1)
    parser.add_argument("--cut_pow", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--perceptor_names", nargs="+", default=["ViT-B/32"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", default="output/")
    parser.add_argument("--out_name", default=None)
    # fmt: on
    return parser


def main(args):
    side_x, side_y = [int(v) for v in args.size.split(",")]
    out_name = (
        args.out_name
        if args.out_name is not None
        else "_".join([p.replace(" ", "_") for p in args.prompts] + [Path(i).stem for i in args.image_prompts])
        + "_guided_diffusion"
    )
    imgs = guided_diffusion(
        prompts=args.prompts,
        image_prompts=args.image_prompts,
        init_image=args.init_image,
        side_x=side_x,
        side_y=side_y,
        diffusion_model_size=args.diffusion_model_size,
        batch_size=args.batch_size,
        clip_guidance_scale=args.clip_guidance_scale,
        tv_scale=args.tv_scale,
        range_scale=args.range_scale,
        init_scale=args.init_scale,
        diffusion_steps=args.diffusion_steps,
        timestep_respacing=args.timestep_respacing,
        skip_timesteps=args.skip_timesteps,
        cutn=args.cutn,
        cutn_batches=args.cutn_batches,
        cut_pow=args.cut_pow,
        device=args.device,
        perceptor_names=args.perceptor_names,
        seed=args.seed,
        outname=f"{args.out_dir}/{out_name}",
        save_intermediate=True,
        cache_model=False,
    )
    for k, image in enumerate(imgs):
        TF.to_pil_image(image).save(f"{args.out_dir}/{out_name}_{k}.png")


if __name__ == "__main__":
    main(argument_parser().parse_args())
