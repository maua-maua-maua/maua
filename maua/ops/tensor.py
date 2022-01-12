from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL.Image import Image, fromarray
from PIL.Image import open as open_img
from torch import Tensor
from torchvision.transforms.functional import to_tensor


def load_image(im: Union[Tensor, Image, Path, str]):
    return im if isinstance(im, Tensor) else (img2tensor(im) if isinstance(im, Image) else img2tensor(open_img(im)))


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


def tensor2bytes(tensor: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch [C,H,W] tensor to bytes (e.g. for passing to FFMPEG)

    Args:
        tensor (torch.Tensor): Image tensor to convert to UINT8 bytes

    Returns:
        np.ndarray
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

