from pathlib import Path
from typing import List

import numpy as np
from PIL.Image import Image, fromarray, open as open_img
from torch import Tensor
from torchvision.transforms.functional import to_tensor


def img2tensor(pil_image):
    return to_tensor(pil_image).unsqueeze(0)


def tensor2img(tensor):
    return fromarray(tensor.squeeze(0).permute(1, 2, 0).mul(255).round().detach().cpu().numpy().astype(np.uint8))


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
