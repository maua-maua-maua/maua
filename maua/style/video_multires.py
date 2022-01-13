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
    sizes=[128, 256, 1024],
    parameterization="rgb",
    perceptor="kbc-vgg19",
    perceptor_kwargs={},
    optimizer="LBFGS",
    lr=0.25,
    optimizer_kwargs={},
    flow_models=["farneback"],
    n_iters=[512, 384, 128],
    passes_per_scale=8,
    blend_factor=1,
    content_weight=0.01,
    style_weight=100,
    tv_weight=10,
    temporal_weight=0.5,
    style_scale=1,
    start_random_frame=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_intermediate=False,
    fps=24,
):
    if isinstance(n_iters, int):
        n_iters = [n_iters] * len(sizes)

    video = None
    for size, iters in zip(sizes, n_iters):
        video = transfer(
            content_video=content_video,
            style_imgs=style_imgs,
            init_video=video,
            init_type="content" if size == sizes[0] else "init_video",
            match_hist=match_hist,
            size=size,
            parameterization=parameterization,
            perceptor=perceptor,
            perceptor_kwargs=perceptor_kwargs,
            optimizer=optimizer,
            lr=lr,
            optimizer_kwargs=optimizer_kwargs,
            flow_models=flow_models,
            n_iters=iters,
            n_passes=32 if size == sizes[0] else passes_per_scale,
            temporal_loss_after=-1,
            blend_factor=blend_factor,
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight,
            temporal_weight=temporal_weight,
            style_scale=style_scale,
            start_random_frame=start_random_frame,
            device=device,
            save_intermediate=save_intermediate,
            fps=fps,
        )
    return video


if __name__ == "__main__":
    import sys

    out_dir = sys.argv[1]
    content = sys.argv[2]
    styles = sys.argv[3:]
    output_name = out_dir + "/" + "_".join([Path(v).stem for v in [content] + styles]) + ".mp4"
    transfer_multires(content, styles, save_intermediate=output_name)
