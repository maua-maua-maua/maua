"""
Neural style transfer
"""

import gc
from pathlib import Path
from typing import List, Union

import torch
from ops.image import match_histogram, resample
from ops.loss import tv_loss
from ops.tensor import load_images, tensor2img
from optimizers import load_optimizer, optimizer_choices
from perceptors import Perceptor, load_perceptor
from PIL import Image
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from .parameterization import Parameterization
from .parameterization.rgb import RGB


@torch.no_grad()
def transfer(
    content_img: Union[Tensor, Image.Image, str],
    style_imgs: List[Union[Tensor, Image.Image, str]],
    init_img: Union[Tensor, Image.Image, str] = None,
    init_type="content",
    match_hist="avg",
    size=512,
    perceptor="pgg-vgg19",
    perceptor_kwargs={},
    optimizer="LBFGS",
    optimizer_kwargs={},
    n_iters=500,
    content_weight=1,
    style_weight=5000,
    tv_weight=0,
    style_scale=1,
    content_layers: List[int] = None,
    style_layers: List[int] = None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Perform a neural style transfer

    Args:
        content_img (Union[Tensor, Image.Image, str]): Image whose structure will be preserved in output
        style_imgs (List[Union[Tensor, Image.Image, str]]): Images whose style will be apparent in output
        init_img (Union[Tensor, Image.Image, str], optional): Image to initialize optimization with.
        init_type (str, optional): How to initialize the image for optimization. Choices ['content', 'random', 'init_img'].
        match_hist (str, optional): How to match color histogram of intermediate images. Choices ['avg', False].
        size (int, optional): Size of output image.
        perceptor (str, optional): Which perceptor to optimize with. Choices ["pgg-vgg19", "pgg-vgg16", "pgg-prune", "pgg-nyud", "pgg-fcn32s", "pgg-sod", "pgg-nin"].
        perceptor_kwargs (dict, optional): Key word arguments for the Perceptor class.
        optimizer (str, optional): Optimizer to use. For choices see optimizers.py
        optimizer_kwargs (dict, optional): Key word arguments for the optimizer.
        n_iters (int, optional): Number of iterations to optimize for.
        content_weight (int, optional): Strength of content preserving loss. Higher values will lead to outputs which better preserve the content's structure and texture.
        style_weight (int, optional): Strength of style loss. Higher values will lead to outputs which look more like the style images.
        tv_weight (int, optional): Strength of total variation loss. Higher values lead to smoother outputs.
        style_scale (int, optional): Scale of style images relative to output image. Larger scales will make textures from styles larger in the output image.
        content_layers (List[int], optional): Layers in Perceptor network that the content loss will be calculated for. Defaults to None which uses defaults defined in each Perceptor class.
        style_layers (List[int], optional): Layers in Perceptor network that the style loss will be calculated for. Defaults to None which uses defaults defined in each Perceptor class.
        device (torch.device, optional): Device to run on.

    Returns:
        Tensor: Result image
    """
    content_img, style_imgs, init_img = load_images(content_img, style_imgs, init_img)

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

    with torch.enable_grad(), tqdm(total=n_iters, desc=f"Optimizing @ {size}px") as pbar:

        optimizer, n_iters = load_optimizer(optimizer, optimizer_kwargs, n_iters, pastiche.parameters())

        def closure(
            pastiche: Parameterization = pastiche,
            target_embeddings: Tensor = target_embeddings,
            perceptor: Perceptor = perceptor,
            optimizer: Optimizer = optimizer,
        ):
            optimizer.zero_grad()
            img = pastiche()

            loss = perceptor.get_loss(img, target_embeddings)
            if tv_weight > 0:
                loss += tv_weight * tv_loss(img)

            loss.backward()
            pbar.update()
            return loss

        for _ in range(n_iters):
            optimizer.step(closure)

    return pastiche()


def argument_parser():
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser(description=transfer.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content")
    parser.add_argument("--styles", nargs="+")
    parser.add_argument("--init_img", default=None)
    parser.add_argument("--init_type", default="content", choices=['content', 'random', 'init_img'])
    parser.add_argument("--match_hist", default="avg", choices=['avg', False])
    parser.add_argument("--size", default=512)
    parser.add_argument("--perceptor", default="pgg-vgg19", choices=["pgg-vgg19", "pgg-vgg16", "pgg-prune", "pgg-nyud", "pgg-fcn32s", "pgg-sod", "pgg-nin"])
    parser.add_argument("--perceptor_kwargs", default={})
    parser.add_argument("--optimizer", default="LBFGS", choices=['LBFGS','LBFGS-20'] + list(optimizer_choices.keys()))
    parser.add_argument("--optimizer_kwargs", default={})
    parser.add_argument("--n_iters", default=500)
    parser.add_argument("--content_weight", default=1)
    parser.add_argument("--style_weight", default=5000)
    parser.add_argument("--tv_weight", default=0)
    parser.add_argument("--style_scale", default=1)
    parser.add_argument("--content_layers", default=None)
    parser.add_argument("--style_layers", default=None)
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # fmt: on

    return parser


if __name__ == "__main__":
    args = argument_parser().parse_args()
    img = transfer(
        content_img=args.content,
        style_imgs=args.styles,
        init_img=args.init_img,
        init_type=args.init_type,
        match_hist=args.match_hist,
        size=args.size,
        perceptor=args.perceptor,
        perceptor_kwargs=args.perceptor_kwargs,
        optimizer=args.optimizer,
        optimizer_kwargs=args.optimizer_kwargs,
        n_iters=args.n_iters,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        style_scale=args.style_scale,
        content_layers=args.content_layers,
        style_layers=args.style_layers,
        device=args.device,
    )
    tensor2img(img).save(f"output/{'_'.join([Path(arg).stem for arg in [args.content] + args.styles])}.png")
