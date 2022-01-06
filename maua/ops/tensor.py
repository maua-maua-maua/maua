from pathlib import Path
from typing import List, Union

import ffmpeg
import numpy as np
import torch
from PIL.Image import Image, fromarray
from PIL.Image import open as open_img
from torch import Tensor
from torchvision.transforms.functional import to_tensor


def load_images(*inputs):
    results = []
    for maybe_nested_paths_imgs_or_tensors in inputs:
        if maybe_nested_paths_imgs_or_tensors is None:
            results.append(None)
        if isinstance(maybe_nested_paths_imgs_or_tensors, (str, Path)):
            results.append(img2tensor(open_img(maybe_nested_paths_imgs_or_tensors)))
        if isinstance(maybe_nested_paths_imgs_or_tensors, Image):
            results.append(img2tensor(maybe_nested_paths_imgs_or_tensors))
        if isinstance(maybe_nested_paths_imgs_or_tensors, Tensor):
            results.append(maybe_nested_paths_imgs_or_tensors)
        if isinstance(maybe_nested_paths_imgs_or_tensors, List):
            results.append(load_images(*maybe_nested_paths_imgs_or_tensors))
    return results


def img2tensor(pil_image):
    return to_tensor(pil_image).unsqueeze(0)


def tensor2img(tensor):
    return fromarray(
        tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).round().detach().cpu().numpy().astype(np.uint8)
    )


def tensor2bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Converts a PyTorch [C,H,W] tensor to bytes (e.g. for passing to FFMPEG)

    Args:
        tensor (torch.Tensor): Image tensor to convert to UINT8 bytes

    Returns:
        torch.Tensor
    """
    return tensor.permute(1, 2, 0).clamp(0, 1).mul(255).cpu().numpy().astype(np.uint8)


def tensor2imgs(tensor: torch.Tensor, format: str = "RGB") -> List[Image]:
    """Converts a PyTorch [B,C,H,W] tensor to PIL images

    Args:
        tensor (torch.Tensor): Image tensor to be converted
        format (str, optional): PIL image format. Defaults to "RGB"

    Returns:
        PIL.Image
    """
    return [fromarray(tensor2bytes(img), format) for img in tensor]


def write_video(tensor: Union[torch.Tensor, np.ndarray], output_file: str, fps: float=24) -> None:
    """Write a tensor [T,C,H,W] to an mp4 file with FFMPEG.

    Args:
        tensor (Union[torch.Tensor, np.ndarray]): Sequence of images to write
        output_file (str): File to write output mp4 to
        fps (float): Frames per second of output video
    """
    output_size = "x".join(reversed([str(s) for s in tensor.shape[2:]]))

    ffmpeg_proc = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=fps, s=output_size)
        .output(output_file, framerate=fps, vcodec="libx264", preset="slow", v="warning")
        .global_args("-benchmark", "-stats", "-hide_banner")
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )

    for frame in tensor:
        frame = frame if isinstance(frame, torch.Tensor) else torch.from_numpy(frame.copy())
        ffmpeg_proc.stdin.write(tensor2bytes(frame).tobytes())

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
