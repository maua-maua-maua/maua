import random
from functools import reduce

import numpy as np
import torch


def factors(n):
    return np.array(list(set(reduce(list.__add__, ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))))


def round_to_closest_divisor(num, div):
    options = factors(num)
    best_idxs = np.argsort(np.abs(div - options))[:2]
    if options[best_idxs[0]] == div:
        return div
    opt = options[best_idxs[random.choice([0, 1])]]
    return opt


def _perlinterpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin_noise(shape, res, tileable=(True, False, False), interpolant=_perlinterpolant):
    """Generate a 3D tensor of perlin noise.

    Args:
        shape: The shape of the generated tensor (tuple of three ints). This must be a multiple of res.
        res: The number of periods of noise to generate along each axis (tuple of three ints). Note shape must be a multiple of res.
        tileable: If the noise should be tileable along each axis (tuple of three bools). Defaults to (False, False, False).
        interpolant: The interpolation function, defaults to t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A tensor of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    res = tuple(round_to_closest_divisor(shape[r], res[r]) for r in range(len(res)))
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1], 0 : res[2] : delta[2]].astype(np.float32)
    grid = grid.transpose(1, 2, 3, 0) % 1
    grid = torch.from_numpy(grid).cuda()
    # Gradients
    theta = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1).astype(np.float32)
    phi = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1).astype(np.float32)
    gradients = np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=3)
    if tileable[0]:
        gradients[-1, :, :] = gradients[0, :, :]
    if tileable[1]:
        gradients[:, -1, :] = gradients[:, 0, :]
    if tileable[2]:
        gradients[:, :, -1] = gradients[:, :, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    gradients = torch.from_numpy(gradients).cuda()
    g000 = gradients[: -d[0], : -d[1], : -d[2]]
    g100 = gradients[d[0] :, : -d[1], : -d[2]]
    g010 = gradients[: -d[0], d[1] :, : -d[2]]
    g110 = gradients[d[0] :, d[1] :, : -d[2]]
    g001 = gradients[: -d[0], : -d[1], d[2] :]
    g101 = gradients[d[0] :, : -d[1], d[2] :]
    g011 = gradients[: -d[0], d[1] :, d[2] :]
    g111 = gradients[d[0] :, d[1] :, d[2] :]
    # Ramps
    n000 = torch.sum(torch.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g000, 3)
    n100 = torch.sum(torch.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g100, 3)
    n010 = torch.sum(torch.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g010, 3)
    n110 = torch.sum(torch.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g110, 3)
    n001 = torch.sum(torch.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g001, 3)
    n101 = torch.sum(torch.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g101, 3)
    n011 = torch.sum(torch.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g011, 3)
    n111 = torch.sum(torch.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g111, 3)
    # Interpolation
    t = interpolant(grid)
    n00 = n000 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n100
    n10 = n010 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n110
    n01 = n001 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n101
    n11 = n011 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n111
    n0 = (1 - t[:, :, :, 1]) * n00 + t[:, :, :, 1] * n10
    n1 = (1 - t[:, :, :, 1]) * n01 + t[:, :, :, 1] * n11
    perlin = (1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1
    perlin = perlin * 2 - 1  # stretch from -1 to 1
    return perlin.cpu()
