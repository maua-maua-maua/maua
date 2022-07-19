"""
file - test.py
Simple quick script to evaluate model on test images.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import os

import torch
import torchvision.models as models
import torchvision.transforms.functional as tvtf

from ..submodules.NIMA.model.model import NIMA
from ..utility import download

global nima_model
nima_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nima_model(checkpoint="modelzoo/nima_epoch34.pth"):
    if not os.path.exists(checkpoint):
        download("", checkpoint)  # gdown --id 1w9Ig_d6yZqUZSR63kPjZLrEjJ1n845B_
    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model)
    model.load_state_dict(torch.load(checkpoint))
    model = model.to(device).eval()
    return model


@torch.inference_mode()
def nima_score(image, ten_crop=False):
    global nima_model
    if nima_model is None:
        nima_model = load_nima_model()

    assert image.min() >= 0 and image.max() <= 1, "Image inputs should be between 0 and 1"

    def preprocess(x):
        x = tvtf.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return torch.stack(
            [tvtf.center_crop(tvtf.resize(x, 224), 224)]
            + ([*tvtf.ten_crop(tvtf.resize(x, 448), 224)] if ten_crop else [])
        )

    imt = preprocess(image).to(device)
    idxs = torch.arange(10).to(device)
    out = nima_model(imt)

    means = torch.tensor([torch.sum(tile * idxs).item() for tile in out])
    std = torch.mean(torch.tensor([torch.sum(tile * (idxs - means[t]) ** 2) ** 0.5 for t, tile in enumerate(out)]))
    score = torch.mean(means)

    return score, std


#     parser = argparse.ArgumentParser()
#     parser.add_argument("test_images", type=str, help="path to folder containing images")
#     parser.add_argument("--model", type=str, help="path to pretrained model", default="modelzoo/nima_epoch34.pth")
#     parser.add_argument("--workers", type=int, default=4, help="number of workers")
#     # vvv   adding extra crops changes focus to quality of all parts of image versus only global quality   vvv
#     parser.add_argument("--ten_crop", action="store_true", help="Whether to also average over ten-way crop of image")
#     args = parser.parse_args()
