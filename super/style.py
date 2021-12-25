from typing import  List

import torch
from PIL import Image
from torchvision.transforms.functional import center_crop, ten_crop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stylescale(images: List[Image.Image]):
    from style_crowsonkb import StyleTransfer

    upscaler = StyleTransfer(devices=["cuda"])

    def callback(sti):
        if sti.i % 100 == 0:
            print(sti)

    for img in images:
        yield upscaler.stylize(
            content_image=img,
            style_images=[
                center_crop(img, min(img.size)),
                *ten_crop(img, min(img.size) // 2),
                *ten_crop(img, max(img.size) // 2),
            ],
            min_scale=max(img.size),
            end_scale=max(img.size) * 4,
            style_size=min(img.size) // 2,
            callback=callback,
            tv_weight=5,
        )
