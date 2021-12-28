import gc
from pathlib import Path

import torch
from ops.image import match_histogram, resample
from ops.tensor import img2tensor, tensor2img
from optimizers import load_optimizer
from perceptors import Perceptor, load_perceptor
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from .parameterization import Parameterization
from .parameterization.rgb import RGB


@torch.no_grad()
def transfer(
    content_img,
    style_imgs,
    init_img,
    init_type,
    match_hist,
    size,
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
    content_img = resample(content_img.to(device), size)
    style_imgs = [resample(im.to(device), size * style_scale) for im in style_imgs]
    content_img = match_histogram(content_img, style_imgs, mode=match_hist)

    if init_img is not None:
        init_tensor = init_img
    elif init_type == "content":
        init_tensor = content_img
    elif init_type == "random":
        init_tensor = None
    pastiche = RGB(content_img.shape[2], content_img.shape[3], tensor=init_tensor).to(device)

    perceptor = load_perceptor(perceptor)(
        content_layers,
        style_layers,
        content_strength=content_weight,
        style_strength=style_weight,
        **perceptor_kwargs,
    ).to(device)
    target_embeddings = perceptor.get_target_embeddings(content_img, style_imgs)

    del content_img, style_imgs
    gc.collect()
    torch.cuda.empty_cache()

    with torch.enable_grad(), tqdm(total=num_iters, desc=f"Optimizing @ {size}px") as pbar:

        optimizer, num_iters = load_optimizer(optimizer, optimizer_kwargs, num_iters, pastiche.parameters())

        def closure(
            pastiche: Parameterization = pastiche,
            target_embeddings: Tensor = target_embeddings,
            perceptor: Perceptor = perceptor,
            optimizer: Optimizer = optimizer,
        ):
            optimizer.zero_grad()
            loss = perceptor.get_loss(pastiche(), target_embeddings)
            loss.backward()
            pbar.update()
            return loss

        for _ in range(num_iters):
            optimizer.step(closure)

    return pastiche()


if __name__ == "__main__":
    import sys

    from PIL import Image

    img = transfer(
        content_img=img2tensor(Image.open(sys.argv[1])),
        style_imgs=[img2tensor(Image.open(path)) for path in sys.argv[2:]],
        init_img=None,
        init_type="content",
        match_hist="avg",
        size=512,
        perceptor="pgg-vgg19",
        perceptor_kwargs={},
        optimizer="lbfgs20",
        optimizer_kwargs={},
        num_iters=500,
        content_weight=1,
        content_layers=[26],
        style_weight=10000,
        style_layers=[3, 8, 17, 26, 35],
        style_scale=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    tensor2img(img).save(f"output/{'_'.join([Path(arg).stem for arg in sys.argv[1:]])}.png")
