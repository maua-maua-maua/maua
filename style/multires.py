import gc
import sys
from pathlib import Path

import torch
from ops.image import match_histogram, resample
from ops.tensor import img2tensor, tensor2img
from PIL import Image

from .transfer import transfer


@torch.no_grad()
def transfer_multires(
    content_img,
    style_imgs,
    init_img,
    init_type,
    match_hist,
    sizes,
    perceptor,
    perceptor_kwargs,
    optimizer,
    optimizer_kwargs,
    num_iters,
    content_weight,
    content_layers,
    style_weight,
    style_layers,
    style_scale,
    device,
):
    if isinstance(num_iters, int):
        num_iters = [num_iters] * len(sizes)

    if init_img is not None:
        img = init_img
    elif init_type == "content":
        img = content_img
    elif init_type == "random":
        img = torch.empty((1, 3, sizes[0], sizes[0]), device=device).uniform_()

    for size, iters in zip(sizes, num_iters):
        img = resample(img, size)
        img = transfer(
            content_img=content_img,
            style_imgs=style_imgs,
            init_img=img,
            init_type="init_img",
            match_hist=match_hist,
            size=size,
            perceptor=perceptor,
            perceptor_kwargs=perceptor_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            num_iters=iters,
            content_weight=content_weight,
            content_layers=content_layers,
            style_weight=style_weight,
            style_layers=style_layers,
            style_scale=style_scale,
            device=device,
        )
        img = match_histogram(img, style_imgs, match_hist)
        gc.collect()
        torch.cuda.empty_cache()
    return img


if __name__ == "__main__":
    img = transfer_multires(
        content_img=img2tensor(Image.open(sys.argv[1])),
        style_imgs=[img2tensor(Image.open(path)) for path in sys.argv[2:]],
        init_img=None,
        init_type="content",
        match_hist="avg",
        sizes=[512, 724, 1024, 1448, 2048],
        perceptor="pgg-vgg19",
        perceptor_kwargs={},
        optimizer="LBFGS",
        optimizer_kwargs=dict(tolerance_grad=-1.0, tolerance_change=-1.0, history_size=100),
        num_iters=[500, 400, 300, 200, 100],
        content_weight=1,
        content_layers=[26],
        style_weight=5000,
        style_layers=[3, 8, 17, 26, 35],
        style_scale=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    tensor2img(img).save(f"output/{'_'.join([Path(arg).stem for arg in sys.argv[1:]])}.png")
