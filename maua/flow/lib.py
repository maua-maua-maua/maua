import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from decord import VideoReader
from npy_append_array import NpyAppendArray as NpyFile
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from maua.flow import check_consistency, motion_edge, resample_flow
from maua.flow.utils import flow_to_image
from maua.ops.video import write_video
from maua.ops.image import scaled_height_width


def flow_warp_map(
    raw_flow: Union[torch.Tensor, np.ndarray],
    size: Union[int, Tuple[int, int]],
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    if isinstance(raw_flow, np.ndarray):
        raw_flow = torch.from_numpy(raw_flow.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)

    if isinstance(size, int):
        h, w, _ = raw_flow.shape
        h, w = scaled_height_width(h, w, size)
    else:
        h, w = size

    flow = gaussian_blur(raw_flow, kernel_size=3)
    flow = flow.squeeze(0).permute(1, 2, 0)
    flow = resample_flow(flow, (h, w))
    flow[..., 0] /= w
    flow[..., 1] /= h

    neutral = torch.stack(torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"), axis=2)
    warp_map = neutral[..., [1, 0]].to(flow) + flow[..., [1, 0]]

    return warp_map.unsqueeze(0)


def get_consistency_map(forward_flow, backward_flow, consistency="full"):
    if consistency == "magnitude":
        reliable_flow = np.sqrt(forward_flow[..., 0] ** 2 + forward_flow[..., 1] ** 2)
    elif consistency == "motion":
        reliable_flow = (
            motion_edge(
                torch.from_numpy(forward_flow.copy()).permute(2, 1, 0).unsqueeze(0),
                torch.from_numpy(backward_flow.copy()).permute(2, 1, 0).unsqueeze(0),
            )
            .numpy()
            .squeeze()
        )
    elif consistency == "full":
        reliable_flow = check_consistency(forward_flow, backward_flow)
    else:
        reliable_flow = np.ones((forward_flow.shape[0], forward_flow.shape[1]))
    return reliable_flow


@torch.inference_mode()
def preprocess_optical_flow(video_file, flow_model, consistency="full", debug_optical_flow=False):
    frf = f"workspace/{Path(video_file).stem}_content.npy"
    fwf = f"workspace/{Path(video_file).stem}_forward_flow.npy"
    bkf = f"workspace/{Path(video_file).stem}_backward_flow.npy"
    rlf = f"workspace/{Path(video_file).stem}_reliable_flow.npy"

    if not (os.path.exists(frf) and os.path.exists(fwf) and os.path.exists(bkf)):
        with NpyFile(frf) as frames, NpyFile(fwf) as forward, NpyFile(bkf) as backward:

            vr = VideoReader(video_file)
            for i in tqdm(range(len(vr)), desc="Estimating optical flow..."):
                frame1 = torch.from_numpy(vr[i].asnumpy()).div(255).permute(2, 0, 1).unsqueeze(0)
                frame2 = torch.from_numpy(vr[(i + 1) % len(vr)].asnumpy()).div(255).permute(2, 0, 1).unsqueeze(0)

                forward_flow = flow_model(frame1, frame2)
                backward_flow = flow_model(frame2, frame1)

                frames.append(np.ascontiguousarray(frame1[None].permute(0, 3, 1, 2).numpy()))
                forward.append(np.ascontiguousarray(forward_flow[None].astype(np.float32)))
                backward.append(np.ascontiguousarray(backward_flow[None].astype(np.float32)))

    forward = np.load(fwf, mmap_mode="r")
    backward = np.load(bkf, mmap_mode="r")
    frames = np.load(frf, mmap_mode="r")

    if not os.path.exists(rlf):
        with NpyFile(rlf) as reliable:
            for forward_flow, backward_flow in zip(forward, backward):
                reliable_flow = get_consistency_map(forward_flow, backward_flow, consistency)
                reliable.append(np.ascontiguousarray(reliable_flow[None].astype(np.float32)))

    reliable = np.load(rlf, mmap_mode="r")

    if debug_optical_flow:
        print("                  ", "min     ", "mean     ", "max     ", "shape")
        print("forward flow (px):", forward.min(), forward.mean(), forward.max(), forward.shape)
        write_video(
            torch.stack([torch.from_numpy(flow_to_image(f)) for f in forward.copy()]).permute(0, 3, 1, 2).div(255),
            f"output/{Path(video_file).stem}_forward_flow.mp4",
        )
        print("backward flow (px):", backward.min(), backward.mean(), backward.max(), backward.shape)
        write_video(
            torch.stack([torch.from_numpy(flow_to_image(f)) for f in backward.copy()]).permute(0, 3, 1, 2).div(255),
            f"output/{Path(video_file).stem}_backward_flow.mp4",
        )
        print("reliable flow (0,1):", reliable.min(), reliable.mean(), reliable.max(), reliable.shape)
        write_video(
            torch.from_numpy(reliable.copy()).unsqueeze(1).tile(1, 3, 1, 1),
            f"output/{Path(video_file).stem}_reliable_flow.mp4",
        )

    return frames, forward, backward, reliable
