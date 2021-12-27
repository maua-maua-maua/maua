import numpy as np
from PIL.Image import fromarray
from torchvision.transforms.functional import to_tensor


def img2tensor(pil_image):
    return to_tensor(pil_image).unsqueeze(0)


def tensor2img(tensor):
    return fromarray(tensor.squeeze(0).permute(1, 2, 0).mul(255).round().detach().cpu().numpy().astype(np.uint8))
