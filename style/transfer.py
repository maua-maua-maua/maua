import gc

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
    height,
    width,
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
    content_img = resample(img2tensor(content_img).to(device), (height, width))
    style_imgs = [resample(img2tensor(im).to(device), min(height, width) * style_scale) for im in style_imgs]
    content_img = match_histogram(content_img, style_imgs, mode=match_hist)

    if init_img is not None:
        init_tensor = img2tensor(init_img)
    elif init_type == "content":
        init_tensor = content_img
    elif init_type == "random":
        init_tensor = None
    pastiche = RGB(height, width, tensor=init_tensor).to(device)

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

    optimizer = load_optimizer(optimizer)(pastiche.parameters(), **optimizer_kwargs)

    with torch.enable_grad(), tqdm(total=num_iters) as pbar:

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

        optimizer.step(closure)

    return pastiche().cpu()


if __name__ == "__main__":
    import sys

    from PIL import Image

    img = transfer(
        content_img=Image.open(sys.argv[1]),
        style_imgs=[Image.open(path) for path in sys.argv[2:]],
        init_img=None,
        init_type="content",
        match_hist="avg",
        height=512,
        width=512,
        perceptor="pgg-vgg19",
        perceptor_kwargs={},
        optimizer="lbfgs",
        optimizer_kwargs=dict(
            max_iter=1000,
            tolerance_change=float(-1),
            tolerance_grad=float(-1),
            history_size=100,
        ),
        num_iters=1000,
        content_weight=1,
        content_layers=[26],
        style_weight=1000,
        style_layers=[3, 8, 17, 26, 35],
        style_scale=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    tensor2img(img).save("output/test.png")
