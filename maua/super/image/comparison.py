import argparse
from pathlib import Path
from time import time

import torch
from tqdm import tqdm
from maua.ops.tensor import tensor2img

from maua.super.image.single import MODEL_MODULES, MODEL_NAMES


def main(args):
    times = {}
    pbar = tqdm(MODEL_NAMES)
    for model_name in pbar:
        pbar.set_description(model_name)
        module = MODEL_MODULES[model_name]
        model = module.load_model(
            model_name.replace("RealESRGAN-", "").replace("SwinIR-", "").replace("waifu2x", "upconv"),
            torch.device(args.device),
        )
        t = time()
        for img, path in zip(module.upscale(args.images, model), args.images):
            tensor2img(img).save(f"{args.out_dir}/{Path(path).stem}_{model_name}.png")
        times[model_name] = (time() - t) / len(args.images)

    print("Average time taken:")
    for k, v in times.items():
        print(k.ljust(35), f"{v:.4f} sec".rjust(20))


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="output/")
    return parser
