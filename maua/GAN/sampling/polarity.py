import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate
from torchvision.transforms import Normalize
from tqdm import tqdm

from ..wrappers.stylegan3 import StyleGAN3, StyleGAN3Mapper, StyleGAN3Synthesizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import clip

CLIP, preprocess = clip.load("ViT-B/32", jit=True)
normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


@torch.no_grad()
def calculate_svds(G, cache_file, N=1600):
    if not os.path.exists(cache_file):
        latents, svds = [], []
        for _ in tqdm(range(N)):

            zs = torch.randn((1, G.z_dim), device=device)
            jacobian = torch.autograd.functional.jacobian(
                lambda z: CLIP.encode_image(normalize(interpolate(G(z), size=224).add(1).div(2))), zs
            )
            # jacobian = torch.autograd.functional.jacobian(lambda z: interpolate(G(z), scale_factor=1 / 64), zs)
            jacobian = jacobian.view(-1, G.z_dim)
            svs = torch.linalg.svdvals(jacobian)

            latents.append(zs.detach().cpu())
            svds.append(svs.detach().cpu())
        latents, svds = np.concatenate(latents), np.stack(svds)
        np.savez(cache_file, latents=latents, svds=svds)

    else:
        with np.load(cache_file) as data:
            latents, svds = data["latents"], data["svds"]

    return torch.from_numpy(latents), torch.from_numpy(svds)


def get_polarity_samples(zs, latents, svds, pol=0, top_k=30, seed=0):
    detz = np.exp(np.log(svds[:, :top_k]).sum(1))
    proba = detz**pol
    proba = np.clip(proba, 1e-60, 1e200)
    idx = np.RandomState(seed).choice(latents.shape[0], size=num_samples, p=proba / proba.sum(), replace=False)
    return latents[idx, :]


def generate(G, z):
    z = torch.from_numpy(z).to(device)
    img = G(z)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img.cpu().numpy()


def polarity_sampling(G, zs, polarity, cache_file, top_k=10):
    latents, svds = calculate_svds(G, cache_file)
    z = get_polarity_samples(zs, latents, svds, polarity, top_k=top_k, seed=None)
    imgs = generate(z)
    return imgs


if __name__ == "__main__":
    import sys

    checkpoint = sys.argv[1]
    G = StyleGAN3(checkpoint)

    grid = []
    for pol in [-2, -1, -0.2, -0.1, 0, 0.01, 0.1, 0.2, 0.5]:
        imgs = polarity_sampling(G=G, polarity=pol, cache_file=f"cache/{Path(checkpoint).stem}_svds.npz")
        imgs = np.concatenate(imgs, axis=1)
        grid.append(imgs)
    grid = np.concatenate(grid, axis=2)
    Image.fromarray(grid).save(f"output/{Path(checkpoint).stem}_polarity_samples.png")
