import argparse
from pathlib import Path
from typing import Generator

import torch
from decord import VideoReader

from maua.ops.video import VideoWriter

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


def interpolate(
    video_file,
    model_name="RIFE-2.3",
    factor=2,
    fp16=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Generator[torch.Tensor, None, None]:

    if fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    module = MODEL_MODULES[model_name]
    model = module.load_model(model_name, device)
    for frame in module.interpolate(video_file, model, factor, fp16):
        yield frame

    if fp16:
        torch.set_default_tensor_type(torch.FloatTensor)


def main(args):
    for video_file in args.video_files:
        vr = VideoReader(video_file)
        fps = vr.get_avg_fps()
        h, w, _ = vr[0].shape

        with VideoWriter(
            output_file=f"{args.out_dir}/{Path(video_file).stem}_{args.model_name}.mp4",
            output_size=(w, h),
            fps=fps * (args.factor / args.slow_factor),
        ) as video:
            for frame in interpolate(video_file, args.model_name, args.factor, not args.no_fp16, args.device):
                video.write(frame.mul(255).round().byte().numpy().tobytes())


def argument_parser():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("video_files", nargs="+")
    parser.add_argument("--model_name", default="RIFE-2.3", choices=MODEL_NAMES)
    parser.add_argument("--factor", type=int, default=4, help="Factor to increase framerate by")
    parser.add_argument("--slow_factor", type=int, default=2, help="Factor to decrease output framerate by (default halves the quadrupled rate, i.e. half speed with double the framerate)")
    parser.add_argument("--no-fp16", action="store_true", help="FP16 reduces memory usage and increases speed on tensor cores (disable for CPU)")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out_dir", default="output/")
    # fmt: on
    return parser
