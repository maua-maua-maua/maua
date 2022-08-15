import random
from math import ceil

import numpy as np
import torch
from tqdm import tqdm

from ...ops.io import hash

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def jacobian_norm_rejection(G, zs, ratio=0.7, N=10, sigma=1e-3):
    """
    Jacobian-based Truncation from "Learning Disconnected Manifolds: a no GAN’s land" by Tanielian et al.
    https://arxiv.org/abs/2006.04596

    Rejects latents which are in areas of the manifold with a high Jacobian norm.

    Due to the connected nature of a GAN's latent manifold, the only way GANs can avoid sampling from non-realistic
    areas of the manifold is to interpolate very quickly through these areas. This induces large values in the Jacobian
    of these areas. Therefore one can avoid these areas by rejecting samples with a high Jacobian norm.

    Args:
        G (torch.nn.Module): Generator
        zs (torch.Tensor): Number of latents to return
        ratio (float): Ratio of samples to keep, lower ratios more strongly restrict samples to modes of G's manifold
        N (int): Number of samples to use for the approximation of the jacobian norm (from paper: 10 is sufficient)
        sigma (float): Noise std for the approximation of the jacobian norm (from paper: 1e-2 to 1e-4 is a good range)
    """
    d = G.z_dim

    n_extra = ceil(len(zs) / ratio) - len(zs) + 1
    extra_zs = torch.from_numpy(np.random.RandomState(random.seed(hash(zs))).randn(n_extra, d)[1:])
    zs = torch.cat((zs, extra_zs)).to(device)

    G = torch.jit.trace(G, torch.randn((N, d), device=device))
    G = torch.jit.optimize_for_inference(G)

    jacnorms = []
    for z in tqdm(zs[:, None], desc="Jacobian norm rejection sampling... "):
        epsilon = torch.normal(0, sigma**2, size=(N, d), device=device)
        jacnorm = torch.mean(torch.norm(G(z + epsilon) - G(z)) ** 2 / sigma**2)  # approximate jacobian frobenius norm
        jacnorms.append(jacnorm)
    idxs = torch.argsort(torch.tensor(jacnorms))
    return zs[idxs[: ceil(ratio * len(zs))]]
