import numpy as np
import torch
from scipy import interpolate

from . import cache_to_workspace
from .postprocess import gaussian_filter


@cache_to_workspace("onset_weighted")
def onset_weighted(base_latent, onset_latent, onsets):
    return base_latent[None] * (1 - onsets) + onset_latent[None] * onsets[:, None, None]


@cache_to_workspace("chroma_weighted")
def chroma_weighted(latents, chroma):
    """Creates chromagram weighted latent sequence

    Args:
        chroma (torch.tensor): Chromagram
        latents (torch.tensor): Latents (must have same number as number of notes in chromagram)

    Returns:
        torch.tensor: Chromagram weighted latent sequence
    """
    chroma /= chroma.sum(1)
    return (chroma[..., None, None] * latents[None]).sum(1)


@cache_to_workspace("pitch_tracking")
def pitch_tracking(latents, pitch):
    low, high = np.percentile(pitch, 25), np.percentile(pitch, 75)
    pitch -= low
    pitch /= high
    pitch *= len(latents)
    pitch %= len(latents)
    return latents.numpy()[pitch.round().astype(int)]


@cache_to_workspace("loops")
def loops(latents, n_frames, fps, tempo, type="spline"):
    bars_per_sec = tempo / 4 / 60
    duration = n_frames / fps
    n_loops = duration * bars_per_sec
    if type == "spline":
        return spline_loops(latents, n_frames, n_loops)
    else:
        return slerp_loops(latents, n_frames, n_loops)


def eerp(a, b, t):
    return a ** (1 - t) * b ** t


def copeerp(a, b, t):
    return a ** t * (1 - b ** t) / (1 - a ** t + b ** t)


def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d


def slerp(a, b, t):
    """Interpolation along geodesic of n-dimensional unit sphere
    from https://github.com/soumith/dcgan.torch/issues/14#issuecomment-200025792

    Args:
        a (float): Starting value
        b (float): Ending value
        t (float): Value between 0 and 1 representing fraction of interpolation completed

    Returns:
        float: Interpolated value
    """
    omega = np.arccos(np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - t) * a + t * b  # L'Hopital's rule/LERP
    return np.sin((1.0 - t) * omega) / so * a + np.sin(t * omega) / so * b


@cache_to_workspace("slerp_loops")
def slerp_loops(latent_selection, n_frames, n_loops):
    """Get looping latents using geodesic interpolation. Total length of n_frames with n_loops repeats.

    Args:
        latent_selection (th.tensor): Set of latents to loop between (in order)
        n_frames (int): Total length of output looping sequence
        n_loops (int): Number of times to loop
        smoothing (int, optional): Standard deviation of gaussian smoothing kernel. Defaults to 1.
        loop (bool, optional): Whether to return to first latent. Defaults to True.

    Returns:
        th.tensor: Sequence of smoothly looping latents
    """
    latent_selection = np.concatenate([latent_selection, latent_selection[[0]]])

    base_latents = []
    for n in range(len(latent_selection)):
        for val in np.linspace(0.0, 1.0, int(n_frames // max(1, n_loops) // len(latent_selection))):
            base_latents.append(
                torch.from_numpy(
                    slerp(
                        latent_selection[n % len(latent_selection)][0],
                        latent_selection[(n + 1) % len(latent_selection)][0],
                        val,
                    )
                )
            )
    base_latents = torch.stack(base_latents)
    base_latents = gaussian_filter(base_latents, 1)
    base_latents = torch.cat([base_latents] * int(n_frames / len(base_latents)), axis=0)
    base_latents = torch.cat([base_latents[:, None, :]] * 18, axis=1)
    if n_frames - len(base_latents) != 0:
        base_latents = torch.cat([base_latents, base_latents[0 : n_frames - len(base_latents)]])
    return base_latents


@cache_to_workspace("spline_loops")
def spline_loops(latent_selection, n_frames, n_loops):
    """Get looping latents using spline interpolation. Total length of n_frames with n_loops repeats.

    Args:
        latent_selection (th.tensor): Set of latents to loop between (in order)
        n_frames (int): Total length of output looping sequence
        n_loops (int): Number of times to loop
        loop (bool, optional): Whether to return to first latent. Defaults to True.

    Returns:
        th.tensor: Sequence of smoothly looping latents
    """
    original_device = latent_selection.device
    latent_selection = latent_selection.cpu()
    latent_selection = np.concatenate([latent_selection, latent_selection[[0]]])

    x = np.linspace(0, 1, int(n_frames // max(1, n_loops)))
    base_latents = np.zeros((len(x), *latent_selection.shape[1:]))
    for lay in range(latent_selection.shape[1]):
        for lat in range(latent_selection.shape[2]):
            tck = interpolate.splrep(np.linspace(0, 1, latent_selection.shape[0]), latent_selection[:, lay, lat])
            base_latents[:, lay, lat] = interpolate.splev(x, tck)

    base_latents = torch.cat([torch.from_numpy(base_latents).float()] * int(n_frames / len(base_latents)), axis=0)
    if n_frames - len(base_latents) > 0:
        base_latents = torch.cat([base_latents, base_latents[0 : n_frames - len(base_latents)]])
    return base_latents[:n_frames].to(original_device)


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
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1], 0 : res[2] : delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    grid = torch.from_numpy(grid).cuda()
    # Gradients
    theta = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    phi = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
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
    return perlin * 2 - 1  # stretch from -1 to 1
