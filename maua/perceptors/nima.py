"""
file - test.py
Simple quick script to evaluate model on test images.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import argparse
from glob import glob
from numpy import arange

import torch
import torchvision.models as models
import torchvision.transforms.functional as tvtf
from PIL import Image

from maua.submodules.NIMA.model.model import *

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("test_images", type=str, help="path to folder containing images")
    parser.add_argument("--model", type=str, help="path to pretrained model", default="modelzoo/nima_epoch34.pth")
    parser.add_argument("--workers", type=int, default=4, help="number of workers")
    # vvv   adding extra crops changes focus to quality of all parts of image versus only global quality   vvv
    parser.add_argument("--ten_crop", action="store_true", help="Whether to also average over ten-way crop of image")
    args = parser.parse_args()

    with torch.inference_mode():
        base_model = models.vgg16(pretrained=True)
        model = NIMA(base_model)
        model.load_state_dict(torch.load(args.model))  # gdown --id 1w9Ig_d6yZqUZSR63kPjZLrEjJ1n845B_
        model = model.to(device).eval()

        def test_transform(x):
            x = tvtf.normalize(tvtf.to_tensor(x), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            return torch.stack(
                [tvtf.center_crop(tvtf.resize(x, 224), 224)]
                + ([*tvtf.ten_crop(tvtf.resize(x, 672), 224)] if args.ten_crop else [])
            )

        idxs = torch.arange(10).to(device)
        for i, img_path in enumerate(glob(args.test_images + "/*")):
            try:
                im = Image.open(img_path).convert("RGB")
                imt = test_transform(im).to(device)

                out = model(imt)

                means = torch.tensor([torch.sum(tile * idxs).item() for tile in out])
                std = torch.mean(
                    torch.tensor([torch.sum(tile * (idxs - means[t]) ** 2) ** 0.5 for t, tile in enumerate(out)])
                )
                mean = torch.mean(means)

                print(f"mean: {mean.item():.3f} | std: {std.item():.3f} \t\t {img_path}")
            except Exception as e:
                print()
                print(img_path, "failed:", e)
                print()
