from typing import List
import numpy as np
import torch
from PIL import Image
import ffmpeg


def tensor2bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Converts a PyTorch [B,C,H,W] tensor to bytes (e.g. for passing to FFMPEG)

    Args:
        tensor (torch.Tensor): Image tensor to convert to UINT8 bytes

    Returns:
        torch.Tensor
    """
    return tensor.permute(0, 2, 3, 1).mul(127.5).add(128).clamp(0, 255).cpu().numpy().astype(np.uint8)


def tensor2imgs(tensor: torch.Tensor, format: str = "RGB") -> List[Image.Image]:
    """Converts a PyTorch [B,C,H,W] tensor to a PIL image

    Args:
        tensor (torch.Tensor): Image tensor to be converted
        format (str, optional): PIL image format. Defaults to "RGB"

    Returns:
        PIL.Image
    """
    return [Image.fromarray(img, format) for img in tensor2bytes(tensor)]


def write_video(tensor: torch.Tensor, output_file: str, fps: float) -> None:
    """Write a tensor [T,C,H,W] to an mp4 file with FFMPEG.

    Args:
        tensor (torch.Tensor): Sequence of images to write
        output_file (str): File to write output mp4 to
        fps (float): Frames per second of output video
    """
    print(f"writing {tensor.shape[0]} frames...")

    output_size = "x".join(reversed([str(s) for s in tensor.shape[1:-1]]))

    ffmpeg_proc = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=fps, s=output_size)
        .output(output_file, framerate=fps, vcodec="libx264", preset="slow", v="warning")
        .global_args("-benchmark", "-stats", "-hide_banner")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in tensor2bytes(tensor):
        ffmpeg_proc.stdin.write(frame.tobytes())

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
