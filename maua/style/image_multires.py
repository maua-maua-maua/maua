import gc
import sys
from pathlib import Path

import torch
from PIL import Image

from maua.ops.image import match_histogram, resample
from maua.ops.tensor import img2tensor, tensor2img

from .image import transfer


@torch.no_grad()
def transfer_multires(
    content_img,
    style_imgs,
    init_img,
    init_type,
    match_hist,
    sizes,
    parameterization,
    perceptor,
    perceptor_kwargs,
    optimizer,
    lr,
    optimizer_kwargs,
    n_iters,
    content_weight,
    style_weight,
    style_scale,
    device,
):
    if isinstance(n_iters, int):
        n_iters = [n_iters] * len(sizes)

    if init_img is not None:
        img = init_img
    elif init_type == "content":
        img = content_img
    elif init_type == "random":
        img = torch.empty((1, 3, sizes[0], sizes[0]), device=device).uniform_().mul(0.1)

    for size, iters in zip(sizes, n_iters):
        img = resample(img, size)
        img = transfer(
            content_img=content_img,
            style_imgs=style_imgs,
            init_img=img,
            init_type="init_img",
            match_hist=match_hist,
            size=size,
            parameterization=parameterization,
            perceptor=perceptor,
            perceptor_kwargs=perceptor_kwargs,
            optimizer=optimizer,
            lr=lr,
            optimizer_kwargs=optimizer_kwargs,
            n_iters=iters,
            content_weight=content_weight,
            style_weight=style_weight,
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
        parameterization="rgb",
        perceptor="pgg-vgg19",
        perceptor_kwargs={},
        optimizer="LBFGS",
        lr=0.5,
        optimizer_kwargs=dict(tolerance_grad=-1.0, tolerance_change=-1.0, history_size=100),
        n_iters=[500, 400, 300, 200, 100],
        content_weight=1,
        style_weight=5000,
        style_scale=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    tensor2img(img).save(f"output/{'_'.join([Path(arg).stem for arg in sys.argv[1:]])}.png")
