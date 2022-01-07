import argparse
from pathlib import Path
from time import time

import torch
from tqdm import tqdm

from . import bsrgan, latent_diffusion, realesrgan, swinir, waifu

name2module = {
    "latent-diffusion": latent_diffusion,
    "RealESRGAN-x4plus": realesrgan,
    "RealESRGAN-x4plus-anime": realesrgan,
    "RealESRGAN-xsx4-animevideo": realesrgan,
    "RealESRGAN-pbaylies-wikiart": realesrgan,
    "RealESRGAN-pbaylies-hr-paintings": realesrgan,
    "SwinIR-L-DFOWMFC-GAN": swinir,
    "SwinIR-L-DFOWMFC-PSNR": swinir,
    "SwinIR-M-DFO-GAN": swinir,
    "SwinIR-M-DFO-PSNR": swinir,
    "waifu2x-anime-noise0": waifu,
    "waifu2x-anime-noise1": waifu,
    "waifu2x-anime-noise2": waifu,
    "waifu2x-anime-noise3": waifu,
    "waifu2x-photo-noise0": waifu,
    "waifu2x-photo-noise1": waifu,
    "waifu2x-photo-noise2": waifu,
    "waifu2x-photo-noise3": waifu,
    "CARN": waifu,
    "BSRGAN": bsrgan,
    "RealSR": bsrgan,
}


def upscale(images, model_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    module = name2module[model_name]
    model = module.load_model(
        model_name.replace("RealESRGAN-", "").replace("SwinIR-", "").replace("waifu2x", "upconv"), torch.device(device)
    )
    for img in module.upscale(images, model):
        yield img


def main(args):
    module = name2module[args.model_name]
    model = module.load_model(
        args.model_name.replace("RealESRGAN-", "").replace("SwinIR-", "").replace("waifu2x", "upconv"),
        torch.device(args.device),
    )
    for img, path in tqdm(zip(module.upscale(args.images, model), args.images)):
        img.save(f"{args.out_dir}/{Path(path).stem}_{args.model_name}.png")


def comparison(args):
    times = {}
    pbar = tqdm(name2module.keys())
    for model_name in pbar:
        pbar.set_description(model_name)
        module = name2module[model_name]
        model = module.load_model(
            model_name.replace("RealESRGAN-", "").replace("SwinIR-", "").replace("waifu2x", "upconv"),
            torch.device(args.device),
        )
        t = time()
        for img, path in zip(module.upscale(args.images, model), args.images):
            img.save(f"{args.out_dir}/{Path(path).stem}_{model_name}.png")
        times[model_name] = (time() - t) / len(args.images)

    print("Average time taken:")
    for k, v in times.items():
        print(k.ljust(35), f"{v:.4f} sec".rjust(20))


def single_model_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model_name", default="latent-diffusion", choices=name2module.keys())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="output/")
    return parser


def comparison_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="output/")
    return parser


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "upscale",
        parents=[single_model_argument_parser()],
        help="Upscale images using a specific model",
        add_help=False,
    ).set_defaults(func=main)
    subparsers.add_parser(
        "comparison",
        parents=[comparison_argument_parser()],
        help="Run all of the models to compare their outputs",
        add_help=False,
    ).set_defaults(func=comparison)
    # subparsers.add_parser( # TODO
    #     "bulk",
    #     parents=[bulk_argument_parser()],
    #     help="Multi-GPU efficient inference for large batches of images",
    #     add_help=False,
    # ).set_defaults(func=bulk)
    return parser
