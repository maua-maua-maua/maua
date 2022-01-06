import argparse
from glob import glob
from pathlib import Path
from typing import List

import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def upscale(images: List[Image.Image], model: str):
    if model == "BSRGAN" or model == "RealSR_JPEG":
        return bsrgan(images, model_name=model)
    elif model == "RealESRGAN":
        return realesrgan(images)
    elif model == "SwinIR":
        return swinir(images)
    elif model == "StyleScale":
        return stylescale(images)
    elif model == "LatentDiffusion":
        return latentdiffusion(images)
    else:
        raise NotImplementedError(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_dir")
    parser.add_argument("-o", "--out_dir", default="output/")
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        choices=["SwinIR", "RealESRGAN", "BSRGAN", "RealSR_JPEG", "StyleScale", "LatentDiffusion"],
        default=["LatentDiffusion"],
    )
    args = parser.parse_args()

    files = glob(args.in_dir + "*")
    images = [Image.open(f) for f in files]
    for model in args.model:
        for f, large in zip(files, upscale(images, model)):
            large.save(f"{args.out_dir}/{Path(f).stem}_{model}.jpg")
