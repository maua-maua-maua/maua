"""
Based on https://github.com/znah/gitart_kunstformen_ca
"""

import sys

import torch
from tqdm import tqdm

from maua.nca.train import VideoWriter, to_rgb, zoom


def generate(model_file, output_file, num_frames=600, size=256):
    # Scope the default-device change to this call so it doesn't leak into the process.
    prev_device = torch.get_default_device()
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    try:
        ca = torch.load(model_file, weights_only=False)

        with VideoWriter(output_file) as vid, torch.no_grad():
            x = ca.seed(1, size)
            for k in tqdm(range(num_frames)):
                step_n = min(2 ** (k // 30), 32)
                for _ in range(step_n):
                    x[:] = ca(x)
                img = to_rgb(x[0]).permute(1, 2, 0).cpu()
                vid.add(zoom(img, 2))

        return output_file
    finally:
        torch.set_default_device(prev_device)


if __name__ == "__main__":
    generate(model_file=sys.argv[1], output_file=sys.argv[2])
