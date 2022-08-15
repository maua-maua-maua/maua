import os
import shutil
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from npy_append_array import NpyAppendArray as NpyFile
from PIL import Image
from torch import Tensor
from torch.nn.functional import grid_sample
from tqdm import tqdm

from ..flow import flow_warp_map, get_flow_model, preprocess_optical_flow
from ..loss import feature_loss, tv_loss
from ..ops.image import match_histogram, resample, scaled_height_width
from ..ops.io import load_images
from ..ops.video import write_video
from ..optimizers import load_optimizer
from ..parameterizations import load_parameterization
from ..perceptors import load_perceptor


@torch.no_grad()
def transfer(
    content_video: Union[str, Path],
    style_imgs: List[Union[Tensor, Image.Image, str, Path]],
    init_video=None,
    init_type="content",
    match_hist="avg",
    size=512,
    parameterization="rgb",
    perceptor="kbc-vgg19",
    perceptor_kwargs={},
    optimizer="LBFGS",
    lr=0.5,
    optimizer_kwargs={},
    flow_models=["farneback"],
    n_iters=512,
    n_passes=16,
    temporal_loss_after=-1,
    blend_factor=1,
    content_weight=1,
    style_weight=5000,
    tv_weight=10,
    temporal_weight=100,
    style_scale=1,
    start_random_frame=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_intermediate=False,
    fps=24,
):
    """Perform style transfer on a video. Uses optical flow to ensure temporal consistency following Ruder et al."s "Artistic style transfer for videos".

    Args:
        content_video (Union[str, Path]): Path to video file to apply style transfer to.
        style_imgs (List[Union[Tensor, Image.Image, str]]): Images whose style will be apparent in output
        init_video (Tensor, np.ndarray, optional): Video to intialize optimization with.
        init_type (str, optional): How to initialize the image for optimization. Choices ["content", "random", "prev_warped", "init_video"].
        match_hist (str, optional): How to match color histogram of intermediate images. Choices ["avg", False].
        size (int, optional): Size of output image.
        parameterization (str, optional): How to parameterize the image. Choices ["rgb", "vqgan"]
        perceptor (str, optional): Which perceptor to optimize with. Choices ["kbc-vgg19", "pgg-vgg19", "pgg-vgg16", "pgg-prune", "pgg-nyud", "pgg-fcn32s", "pgg-sod", "pgg-nin"].
        perceptor_kwargs (dict, optional): Key word arguments for the Perceptor class.
        optimizer (str, optional): Optimizer to use. For choices see optimizers.py
        lr (float, optional): Optimizer learning rate.
        optimizer_kwargs (dict, optional): Key word arguments for the optimizer.
        flow_models (list, optional): A list of models to use calculate optical flow. Choices ["spynet", "pwc", "liteflownet", "unflow", "farneback"]
        n_iters (int, optional): Number of iterations to optimize for.
        n_passes (int, optional): Number of passes to make over video.
        temporal_loss_after (int, optional): Pass after which to start enforcing temporal consistency.
        blend_factor (float, optional): Factor with which previous frame is blended into intialization of next frame.
        content_weight (int, optional): Strength of content preserving loss. Higher values will lead to outputs which better preserve the content"s structure and texture.
        style_weight (int, optional): Strength of style loss. Higher values will lead to outputs which look more like the style images.
        tv_weight (int, optional): Strength of total variation loss. Higher values lead to smoother outputs.
        temporal_weight (int, optional): Strength of temporal loss. Higher values lead to more consistency between frames.
        style_scale (int, optional): Scale of style images relative to output image. Larger scales will make textures from styles larger in the output image.
        start_random_frame (bool, optional): Start stylization from a random frame. Repeated random starting positions can help ensure that a video loops smoothly.
        content_layers (List[int], optional): Layers in Perceptor network that the content loss will be calculated for. Defaults to None which uses defaults defined in each Perceptor class.
        style_layers (List[int], optional): Layers in Perceptor network that the style loss will be calculated for. Defaults to None which uses defaults defined in each Perceptor class.
        device (torch.device, optional): Device to run on.
        save_intermediate (bool, str, Path, optional): Filename to save intermediate pass videos to.
        fps (float, optional): Framerate for intermediate saved videos.

    Returns:
        np.ndarray: A numpy memmap of the styled video frames
    """
    style_imgs = load_images(*style_imgs)
    style_imgs = [resample(im.to(device), size * style_scale) for im in style_imgs]
    content, forward, backward, reliable = preprocess_optical_flow(content_video, get_flow_model(flow_models))

    perceptor = load_perceptor(perceptor)(
        content_strength=content_weight, style_strength=style_weight, **perceptor_kwargs
    ).to(device)
    style_embeddings = perceptor.get_target_embeddings(contents=None, styles=style_imgs)
    style_imgs = [im.cpu() for im in style_imgs]

    h, w = scaled_height_width(content.shape[2], content.shape[3], size)
    pastiche = load_parameterization(parameterization)(h, w).to(device)

    d = 1
    prev_frame_file = (
        f"workspace/{Path(save_intermediate).stem if save_intermediate else Path(content_video).stem}_frames_prev.npy"
    )
    next_frame_file = (
        f"workspace/{Path(save_intermediate).stem if save_intermediate else Path(content_video).stem}_frames_next.npy"
    )

    with tqdm(total=n_iters * len(content)) as pbar:
        for p_n in range(n_passes):
            pbar.set_description(f"Optimizing @ {size}px pass {p_n + 1} of {n_passes}")

            if os.path.exists(prev_frame_file):
                frames = np.load(prev_frame_file, mmap_mode="r")
            else:
                frames = content

            with NpyFile(next_frame_file) as styled:
                frame_range = list(range(len(content)))
                if start_random_frame:
                    start_idx = np.random.randint(0, len(content))
                    frame_range = frame_range[start_idx:] + frame_range[:start_idx]
                for f_n in frame_range:

                    content_frame = resample(torch.from_numpy(content[[f_n]].copy()).to(device), (h, w))
                    content_embeddings = perceptor.get_target_embeddings(contents=content_frame, styles=None)
                    target_embeddings = torch.cat((content_embeddings, style_embeddings))
                    del content_frame, content_embeddings

                    curr_frame = resample(torch.from_numpy(frames[[f_n]].copy()).to(device), (h, w))
                    prev_frame = resample(torch.from_numpy(frames[[(f_n - d) % len(frames)]].copy()).to(device), (h, w))

                    using_blending = blend_factor > 0 and 0 < p_n < n_passes - 1
                    using_temporal_loss = temporal_weight > 0 and p_n > temporal_loss_after
                    if using_blending or using_temporal_loss or init_type == "prev_warped":
                        flow_map = flow_warp_map((forward if d == 1 else backward)[f_n], (h, w)).to(device)
                        flow_mask = resample(torch.from_numpy(reliable[None, [f_n]].copy()).to(device), (h, w))
                        prev_warped = grid_sample(prev_frame, flow_map, padding_mode="border", align_corners=False)

                    init_tensor = None
                    if init_type == "prev_warped":
                        init_tensor = prev_warped
                    elif p_n == 0:
                        if init_type == "random":
                            init_tensor = torch.empty_like(curr_frame).uniform_().mul(0.1)
                        elif init_type == "init_video" and init_video is not None:
                            init_tensor = init_video[[f_n]]
                            if isinstance(init_tensor, np.ndarray):
                                init_tensor = torch.from_numpy(init_tensor.copy())
                            init_tensor = resample(init_tensor.to(device), (h, w))
                    if init_tensor is None:
                        init_tensor = curr_frame

                    if using_blending:
                        blend_mask = blend_factor * flow_mask
                        init_tensor += blend_mask * prev_warped
                        init_tensor /= 1 + blend_mask

                    init_tensor = match_histogram(init_tensor, style_imgs, match_hist)
                    pastiche.encode(init_tensor)
                    pastiche.reset_ema()
                    del curr_frame, prev_frame, init_tensor

                    with torch.enable_grad():

                        opt, niter = load_optimizer(
                            optimizer, lr, optimizer_kwargs, n_iters // n_passes, pastiche.parameters()
                        )

                        def closure():
                            opt.zero_grad()
                            pastiche.update_ema()

                            img = pastiche()

                            loss = perceptor.get_loss(img, target_embeddings)

                            if tv_weight > 0:
                                loss += tv_weight * tv_loss(img)

                            if using_temporal_loss:
                                img.register_hook(lambda grad: grad * flow_mask)
                                loss += temporal_weight * feature_loss(img, prev_warped)

                            loss.backward()
                            pbar.update()
                            return loss

                        for _ in range(niter):
                            opt.step(closure)

                    result = match_histogram(pastiche.decode_average(), style_imgs, match_hist).detach().cpu().numpy()
                    styled.append(result)

            if save_intermediate:
                write_video(np.load(next_frame_file, mmap_mode="r"), save_intermediate, fps=fps)
            d = -d
            shutil.move(next_frame_file, prev_frame_file)

    return np.load(prev_frame_file, mmap_mode="r")


def main(args):
    if len(args.perceptor_kwargs) > 0:
        perceptor_kwargs = {
            k: eval(t)(v)
            for k, t, v in zip(args.perceptor_kwargs[::3], args.perceptor_kwargs[1::3], args.perceptor_kwargs[2::3])
        }
    else:
        perceptor_kwargs = {}

    if len(args.optimizer_kwargs) > 0:
        optimizer_kwargs = {
            k: eval(t)(v)
            for k, t, v in zip(args.optimizer_kwargs[::3], args.optimizer_kwargs[1::3], args.optimizer_kwargs[2::3])
        }
    else:
        optimizer_kwargs = {}

    output_name = args.out_dir + "/" + "_".join([Path(v).stem for v in [args.content] + args.styles]) + ".mp4"
    video = transfer(
        content_video=args.content,
        style_imgs=args.styles,
        init_type=args.init_type,
        match_hist=args.match_hist,
        size=args.size,
        parameterization=args.parameterization,
        perceptor=args.perceptor,
        perceptor_kwargs=perceptor_kwargs,
        optimizer=args.optimizer,
        lr=args.lr,
        optimizer_kwargs=optimizer_kwargs,
        flow_models=args.flow_models,
        n_iters=args.n_iters,
        n_passes=args.n_passes,
        temporal_loss_after=args.temporal_loss_after,
        blend_factor=args.blend_factor,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        temporal_weight=args.temporal_weight,
        style_scale=args.style_scale,
        start_random_frame=args.start_random_frame,
        device=args.device,
        save_intermediate=output_name,
        fps=args.fps,
    )
    write_video(video, output_name, args.fps)
