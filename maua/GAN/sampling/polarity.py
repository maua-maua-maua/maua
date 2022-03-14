import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..wrappers.stylegan3 import StyleGAN3, StyleGAN3Mapper, StyleGAN3Synthesizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_svds(G, cache_file, batch_size=8, N=1500 * 8):

    if not os.path.exists(cache_file):
        latents, svds = [], []
        for _ in tqdm(range(0, N, batch_size)):

            zs = torch.randn((batch_size, G.mapper.G_map.z_dim), device=device)
            zs.requires_grad_()
            imgs = G(zs)
            noise = torch.randn_like(imgs) / np.sqrt(imgs.shape[2] * imgs.shape[3])
            jacobian = torch.autograd.grad(outputs=[(imgs * noise).sum()], inputs=[zs])[0]

            with torch.inference_mode():
                vs = torch.svd(jacobian).V

            latents.append(zs.detach().cpu())
            svds.append(vs.detach().cpu())
        latents, svds = np.concatenate(latents), np.concatenate(svds)
        np.savez(cache_file, latents, svds)

    else:
        with np.load(cache_file) as data:
            latents, svds = data["latents"], data["svds"]

    return torch.from_numpy(latents), torch.from_numpy(svds)


def get_polarity_samples(num_samples, latents, svds, pol=0, top_k=30, seed=0):
    detz = np.exp(np.log(svds[:, :top_k]).sum(1))
    proba = detz**pol
    proba = np.clip(proba, 1e-60, 1e200)
    np.random.seed(seed)
    idx = np.random.choice(latents.shape[0], size=num_samples, p=proba / proba.sum(), replace=False)
    return latents[idx, :]


def generate(G, z):
    z = torch.from_numpy(z).to(device)
    img = G(z)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img.cpu().numpy()


def polarity_sampling(G, polarity, cache_file, n=20, top_k=10):
    latents, svds = calculate_svds(G, cache_file)
    z = get_polarity_samples(n, latents, svds, polarity, top_k=top_k, seed=None)
    imgs = generate(z)
    return imgs


if __name__ == "__main__":
    import sys

    checkpoint = sys.argv[1]

    G = StyleGAN3(
        StyleGAN3Mapper(checkpoint, False), StyleGAN3Synthesizer(checkpoint, False, (1024, 1024), "stretch", 0)
    ).to(device)

    grid = []
    for pol in [-2, -1, -0.2, -0.1, 0, 0.01, 0.1, 0.2, 0.5]:
        imgs = polarity_sampling(G=G, polarity=pol, cache_file=f"cache/{Path(checkpoint).stem}_svds.npz")
        imgs = np.concatenate(imgs, axis=1)
        grid.append(imgs)
    grid = np.concatenate(grid, axis=2)
    Image.fromarray(grid).save(f"output/{Path(checkpoint).stem}_polarity_samples.png")
