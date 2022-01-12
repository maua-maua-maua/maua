import argparse
from pathlib import Path

import ffmpeg
import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

from maua.super.image import MODEL_NAMES
from maua.super.image import upscale as upscale_images


def upscale(video_file, model_name, device, out_dir):
    vr = VideoReader(video_file, ctx=cpu())
    fps = vr.get_avg_fps()
    h, w, c = vr[0].shape

    ffmpeg_proc = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt=f"rgb{8*c}", framerate=fps, s=f"{4*w}x{4*h}")
        .output(
            f"{out_dir}/{Path(video_file).stem}_{model_name}.mp4",
            framerate=fps,
            vcodec="libx264",
            preset="slow",
            v="warning",
        )
        .global_args("-benchmark", "-stats", "-hide_banner")
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )

    frames = (torch.from_numpy(vr[i].asnumpy()).permute(2, 0, 1).unsqueeze(0).div(255) for i in range(len(vr)))
    for large in tqdm(upscale_images(frames, model_name, device), total=len(vr)):
        ffmpeg_proc.stdin.write(np.asarray(large).tobytes())

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()


def main(args):
    for video_file in args.video_files:
        upscale(video_file, args.model_name, args.device, args.out_dir)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_files", nargs="+")
    parser.add_argument("--model_name", default="latent-diffusion", choices=MODEL_NAMES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="output/")
    return parser
