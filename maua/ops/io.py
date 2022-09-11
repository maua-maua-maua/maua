from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image, fromarray
from PIL.Image import open as open_img
from torch import Tensor
from torchvision.transforms.functional import to_pil_image, to_tensor


def save_image(tensor, filename):
    to_pil_image(tensor.squeeze().add(1).div(2).clamp(0, 1)).save(filename)


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


def img2tensor(pil_image, format: str = "RGB"):
    return to_tensor(pil_image.convert(format)).unsqueeze(0)


def tensor2img(tensor, format: str = "RGB"):
    return fromarray(
        tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().detach().cpu().numpy(), format
    )


def tensor2bytes(tensor: torch.Tensor, value_range: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """Converts a PyTorch [1,C,H,W] tensor to bytes (e.g. for passing to FFMPEG)

    Args:
        tensor (torch.Tensor): Image tensor to convert to UINT8 bytes

    Returns:
        np.ndarray
    """
    mn, mx = value_range
    return (
        tensor.squeeze(0)
        .permute(1, 2, 0)
        .clamp(mn, mx)
        .sub(mn)
        .div(mx - mn)
        .mul(255)
        .round()
        .byte()
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    )


def tensor2imgs(tensor: torch.Tensor, format: str = "RGB") -> List[Image]:
    """Converts a PyTorch [B,C,H,W] tensor to PIL images

    Args:
        tensor (torch.Tensor): Image tensor to be converted
        format (str, optional): PIL image format. Defaults to "RGB"

    Returns:
        PIL.Image
    """
    return [tensor2img(img, format) for img in tensor]


def hash(tensor_array_int_obj):
    if isinstance(tensor_array_int_obj, (np.ndarray, torch.Tensor)):
        if isinstance(tensor_array_int_obj, torch.Tensor):
            array = tensor_array_int_obj.detach().cpu().numpy()
        else:
            array = tensor_array_int_obj
        array = deepcopy(array)
        array = array - array.min()
        array = array / array.max()
        byte_tensor = (array * 255).ravel().astype(np.uint8)
        hash = 0
        for ch in byte_tensor[:1024:4]:
            hash = (hash * 281 ^ ch * 997) & 0xFFFFFFFF
        return str(hex(hash)[2:].upper().zfill(8))
    if isinstance(tensor_array_int_obj, (float, int, str, bool)):
        return str(tensor_array_int_obj)
    return ""
