import torch

torch.cuda.is_available()

# torch MUST be imported before decord for reasons?!

import decord

decord.bridge.set_bridge("torch")

import os
from pathlib import Path
from typing import Generator

from decord import VideoReader
from tqdm import tqdm

from ....ops.video import VideoWriter
from . import rife

MODEL_MODULES = {
    "RIFE-1.0": rife,
    "RIFE-1.1": rife,
    "RIFE-2.0": rife,
    "RIFE-2.1": rife,
    "RIFE-2.2": rife,
    "RIFE-2.3": rife,
    "RIFE-2.4": rife,
    "RIFE-3.0": rife,
    "RIFE-3.1": rife,
    "RIFE-3.2": rife,
    "RIFE-3.4": rife,
    "RIFE-3.5": rife,
    "RIFE-3.6": rife,
    "RIFE-3.8": rife,
    "RIFE-3.9": rife,
    "RIFE-4.0": rife,
}
MODEL_NAMES = list(MODEL_MODULES.keys())


def interpolate_video(
    video_file,
    model_name="RIFE-2.3",
    interpolation_factor=2,
    fp16=True,
    decimate=1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Generator[torch.Tensor, None, None]:

    module = MODEL_MODULES[model_name]
    model = module.load_model(model_name, device, fp16)

    vr = VideoReader(video_file)
    N = len(vr)
    for i in tqdm(range(N)):
        frame1 = vr[i].permute(2, 0, 1).unsqueeze(0).div(255).to(model.device)
        frame2 = vr[(i + 1) % N].permute(2, 0, 1).unsqueeze(0).div(255).to(model.device)
        for f, frame in enumerate(module.interpolate(frame1, frame2, model, interpolation_factor, fp16)):
            if f % decimate == decimate // 2:
                yield frame


def main(args):
    for video_file in args.video_files:
        vr = VideoReader(video_file)
        fps = vr.get_avg_fps()
        h, w, _ = vr[0].shape

        out_file = f"{args.out_dir}/{Path(video_file).stem}_{args.model_name}.mp4"
        if os.path.exists(out_file):
            print(f"Skipping {Path(video_file).stem}, output {Path(out_file).stem} already exists!")
            continue

        with VideoWriter(
            output_file=out_file,
            output_size=(w, h),
            fps=fps * (args.interpolation_factor / args.slower / args.decimate),
        ) as video:
            for frame in interpolate_video(
                video_file, args.model_name, args.interpolation_factor, not args.no_fp16, args.decimate, args.device
            ):
                video.write(frame)
