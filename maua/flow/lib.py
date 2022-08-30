import os
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader
from npy_append_array import NpyAppendArray as NpyFile
from tqdm import tqdm

from ..ops.video import write_video
from .consistency import check_consistency, check_consistency_np
from .utils import flow_to_image

NEUTRAL = None


def flow_warp_map(flow: torch.Tensor) -> torch.Tensor:
    b, h, w, two = flow.shape
    flow[..., 0] /= w
    flow[..., 1] /= h
    global NEUTRAL
    if NEUTRAL is None or (NEUTRAL.shape[1], NEUTRAL.shape[2]) != (h, w):
        NEUTRAL = (
            torch.stack(torch.meshgrid(torch.linspace(-1, 1, w), torch.linspace(-1, 1, h), indexing="xy"), axis=2)
            .unsqueeze(0)
            .to(flow)
        )
    warp_map = NEUTRAL + flow
    return warp_map


def get_consistency_map(forward_flow, backward_flow, consistency="full"):
    if consistency == "magnitude":
        reliable_flow = torch.sqrt(forward_flow[..., 0] ** 2 + forward_flow[..., 1] ** 2)
    elif consistency == "full":
        reliable_flow = check_consistency(forward_flow, backward_flow)
    elif consistency == "numpy":
        reliable_flow = torch.from_numpy(
            check_consistency_np(
                forward_flow.detach().cpu().numpy(),
                backward_flow.detach().cpu().numpy(),
            )
        )
    else:
        reliable_flow = torch.ones((forward_flow.shape[0], forward_flow.shape[1]))
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

                frames.append(np.ascontiguousarray(frame1.cpu().numpy()))
                forward.append(np.ascontiguousarray(forward_flow.cpu().numpy()))
                backward.append(np.ascontiguousarray(backward_flow.cpu().numpy()))

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
