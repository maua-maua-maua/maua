from math import ceil
from typing import Union

import ffmpeg
import numpy as np
import torch

from .image import resample
from .tensor import tensor2bytes


class VideoWriter:
    def __init__(
        self,
        output_file,
        output_size,
        fps: float = 24,
        audio_file=None,
        audio_offset=0,
        audio_duration=None,
        ffmpeg_preset="veryslow",
        debug=False,
    ):
        self.output_file = output_file
        self.output_size = f"{2*ceil(output_size[0]/2)}x{2*ceil(output_size[1]/2)}"
        self.fps = fps
        self.audio_file = audio_file
        self.audio_offset = audio_offset
        self.audio_duration = audio_duration
        self.ffmpeg_preset = ffmpeg_preset
        self.debug = debug

    def write(self, tensor):
        _, _, h, w = tensor.shape
        if h % 2 or w % 2:
            tensor = resample(tensor, (2 * ceil(h / 2), 2 * ceil(w / 2)))
        self.ffmpeg_proc.stdin.write(tensor2bytes(tensor))

    def __enter__(self):
        if self.audio_file is not None:
            audio_kwargs = dict(ss=self.audio_offset, guess_layout_max=0)
            if self.audio_duration is not None:
                audio_kwargs["t"] = self.audio_duration
            audio = ffmpeg.input(self.audio_file, **audio_kwargs)
            self.ffmpeg_proc = (
                ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=self.fps, s=self.output_size)
                .output(
                    audio,
                    self.output_file,
                    framerate=self.fps,
                    pix_fmt="yuv420p",
                    preset=self.ffmpeg_preset,
                    audio_bitrate="320K",
                    ac=2,
                    v="warning",
                )
                .global_args("-hide_banner")
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=not self.debug)
            )
        else:
            self.ffmpeg_proc = (
                ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=self.fps, s=self.output_size)
                .output(
                    self.output_file,
                    framerate=self.fps,
                    pix_fmt="yuv420p",
                    preset=self.ffmpeg_preset,
                    v="warning",
                )
                .global_args("-hide_banner")
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=not self.debug)
            )
        return self

    def __exit__(self, type, value, traceback):
        self.ffmpeg_proc.stdin.close()
        self.ffmpeg_proc.wait()


def write_video(
    tensor: Union[torch.Tensor, np.ndarray],
    output_file: str,
    fps: float = 24,
    audio_file=None,
    audio_offset=0,
    audio_duration=None,
    ffmpeg_preset="slow",
    debug=False,
) -> None:
    """Write a tensor [T,C,H,W] to an mp4 file with FFMPEG.

    Args:
        tensor (Union[torch.Tensor, np.ndarray]): Sequence of images to write
        output_file (str): File to write output mp4 to
        fps (float): Frames per second of output video
    """
    _, _, h, w = tensor.shape
    with VideoWriter(output_file, (w, h), fps, audio_file, audio_offset, audio_duration, ffmpeg_preset, debug) as video:
        for frame in tensor:
            frame = frame if isinstance(frame, torch.Tensor) else torch.from_numpy(frame.copy())
            video.write(frame[None])
