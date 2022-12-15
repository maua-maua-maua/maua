import gc
import os
import shutil
from pathlib import Path
from typing import Generator

import torch
from tqdm import tqdm

from ...ops.io import tensor2img
from .models import bsrgan, latent_diffusion, realesrgan, swinir, waifu


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_MODULES = {
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
MODEL_NAMES = list(MODEL_MODULES.keys())


@torch.inference_mode()
def upscale(
    images, model_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Generator[torch.Tensor, None, None]:

    module = MODEL_MODULES[model_name]
    model = module.load_model(
        model_name.replace("RealESRGAN-", "").replace("SwinIR-", "").replace("waifu2x", "upconv"), torch.device(device)
    )
    model = torch.compile(model)
    for img in module.upscale(images, model):
        yield img


@torch.inference_mode()
def upscale_image(
    image, model_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Generator[torch.Tensor, None, None]:

    module = MODEL_MODULES[model_name]
    model = module.load_model(
        model_name.replace("RealESRGAN-", "").replace("SwinIR-", "").replace("waifu2x", "upconv"), torch.device(device)
    )
    image = [im for im in module.upscale([image], model)][0]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return image


def main(args):
    module = MODEL_MODULES[args.model_name]
    model = module.load_model(
        args.model_name.replace("RealESRGAN-", "").replace("SwinIR-", "").replace("waifu2x", "upconv"),
        torch.device(args.device),
    )
    imgiter = iter(module.upscale(args.images, model))
    for p, path in enumerate(tqdm(args.images)):
        out_path = f"{args.out_dir}/{Path(path).stem}_{args.model_name}.png"
        if os.path.exists(out_path):
            continue
        try:
            img = next(imgiter)
            im = tensor2img(img.cpu().float())
            if args.postdownsample > 1:
                im = im.resize((im.size[0] // args.postdownsample, im.size[1] // args.postdownsample))
            im.save(out_path)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                shutil.copy(path, out_path)
                imgiter = iter(module.upscale(args.images[p + 1 :], model))
            else:
                raise
