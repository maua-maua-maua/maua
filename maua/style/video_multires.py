from pathlib import Path
from typing import List, Union

import torch
from PIL import Image
from torch import Tensor

from .video import transfer


@torch.no_grad()
def transfer_multires(
    content_video: Union[str, Path],
    style_imgs: List[Union[Tensor, Image.Image, str, Path]],
    match_hist="avg",
    sizes=[128, 256, 512, 724, 1024],
    perceptor="pgg-vgg19",
    perceptor_kwargs={},
    optimizer="LBFGS",
    optimizer_kwargs={},
    flow_models=["farneback"],
    n_iters=[256, 224, 192, 160, 128],
    passes_per_scale=8,
    blend_factor=1,
    content_weight=1,
    style_weight=5000,
    tv_weight=1,
    temporal_weight=1,
    style_scale=1,
    content_layers: List[int] = None,
    style_layers: List[int] = None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_intermediate=False,
    fps=25,
):
    if isinstance(n_iters, int):
        n_iters = [n_iters] * len(sizes)

    video = None
    for size, iters in zip(sizes, n_iters):
        video = transfer(
            content_video=content_video,
            style_imgs=style_imgs,
            init_video=video,
            init_type="init_video",
            match_hist=match_hist,
            size=size,
            perceptor=perceptor,
            perceptor_kwargs=perceptor_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            flow_models=flow_models,
            n_iters=iters,
            n_passes=passes_per_scale,
            temporal_loss_after=0,
            blend_factor=blend_factor,
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight,
            temporal_weight=temporal_weight,
            style_scale=style_scale,
            content_layers=content_layers,
            style_layers=style_layers,
            device=device,
            save_intermediate=save_intermediate,
            fps=fps,
        )
    return video


if __name__ == "__main__":
    import sys

    content = sys.argv[1]
    styles = sys.argv[2:]
    output_name = "output/" + "_".join([Path(v).stem for v in [content] + styles]) + ".mp4"
    transfer_multires(content, styles, save_intermediate=output_name)
