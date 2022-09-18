from typing import Tuple

import numpy as np
import torch
from torch.nn.functional import conv1d, pad
from torchaudio.functional import contrast, highpass_biquad, lowpass_biquad

from .efficient_quantile import quantile


def gaussian_filter(x, sigma, mode: str = "circular", causal: float = 1):
    dim = len(x.shape)
    n_frames = x.shape[0]
    while len(x.shape) < 3:
        x = x[:, None]

    radius = min(int(sigma * 4), 3 * len(x))
    channels = x.shape[1]

    kernel = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-0.5 / sigma**2 * kernel**2)
    # kernel[radius + 1 :] = kernel[radius + 1 :] * causal  # make kernel less responsive to future information
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)

    if radius > n_frames:  # prevent padding errors on short sequences
        x = pad(x, (n_frames, n_frames), mode=mode)
        print(
            f"WARNING: Gaussian filter radius ({int(sigma * 4)}) is larger than number of frames ({n_frames}).\n\t Filter size has been lowered to ({radius}). You might want to consider lowering sigma ({sigma})."
        )
        x = pad(x, (radius - n_frames, radius - n_frames), mode="replicate")
    else:
        x = pad(x, (radius, radius), mode=mode)

    x = conv1d(x, weight=kernel, groups=channels)

    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t, c, h, w)

    if len(x.shape) > dim:
        x = x.squeeze()

    return x


# @torch.jit.script
def normalize(array):
    array = array - array.min()
    array = array / (array.max() + 1e-8)
    return array


def standardize(array):
    result = torch.clamp(array, quantile(array, 0.25), quantile(array, 0.75) + 1e-10)
    result = normalize(result)
    return result


def cart2pol(x, y):
    if isinstance(x, np.ndarray):
        rho = np.sqrt(np.square(x) + np.square(y))
        phi = np.arctan2(y, x)
    else:
        rho = torch.sqrt(torch.square(x) + torch.square(y))
        phi = torch.atan2(y, x)
    return rho, phi


def median_filter2d(
    x,
    k: Tuple[int, int] = (3, 3),
    s: Tuple[int, int] = (1, 1),
    p: Tuple[int, int, int, int] = (1, 1, 1, 1),
    mode: str = "reflect",
):
    x = pad(x, p, mode=mode)
    x = x.unfold(2, k[0], s[0]).unfold(3, k[1], s[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x


# @torch.jit.script
def spectral_flux(spec):
    return torch.diff(spec, dim=0, append=torch.zeros((1, spec.shape[1]), device=spec.device))


# @torch.jit.script
def onset_envelope(flux):
    u = torch.sum(0.5 * (flux + torch.abs(flux)), dim=1)
    u = torch.clamp(u, quantile(u, 0.025), quantile(u, 0.975))
    u -= u.min()
    u /= u.max()
    return u


def clamp_peaks_percentile(signal, percent):
    if len(signal.shape) < 2:
        signal = signal.unsqueeze(1)

    result = []
    for sig in signal.unbind(1):
        locs = torch.arange(0, sig.shape[0], device=sig.device)
        peaks = torch.ones(sig.shape, dtype=bool, device=sig.device)

        main = sig[locs]
        plus = sig[(locs + 1).clamp(0, sig.shape[0] - 1)]
        minus = sig[(locs - 1).clamp(0, sig.shape[0] - 1)]

        peaks &= torch.gt(main, plus)
        peaks &= torch.gt(main, minus)

        sig = torch.clamp(sig, None, torch.quantile(sig[peaks], percent / 100))

        result.append(sig)

    return torch.stack(result, dim=1)


def clamp_upper_percentile(signal, percentile):
    return torch.clamp(signal, None, torch.quantile(signal, percentile / 100, dim=0))


def clamp_lower_percentile(signal, percentile):
    return torch.clamp(signal, torch.quantile(signal, percentile / 100, dim=0), None)


def emphasize(envs, strength, percentile):
    min = envs.min(dim=0).values
    x = envs - min
    max = x.max(dim=0).values
    x = x / max
    x = x * (1 + torch.tanh(strength * (x - torch.quantile(x, q=percentile / 100, dim=0))))
    return (x * max) + min


def low_pass(audio, sr, fmax=200):
    return lowpass_biquad(audio, sr, fmax)


def mid_pass(audio, sr, fmin=200, fmax=4000):
    return low_pass(high_pass(audio, sr, fmax), sr, fmin)


def high_pass(audio, sr, fmin=4000):
    return highpass_biquad(audio, sr, fmin)


def contrast_enhance(audio, sr, strength=75):
    return contrast(audio, sr, strength)


def confusion_matrix(target, prediction, num_classes):
    unique_mapping = (target.view(-1) * num_classes + prediction.view(-1)).to(torch.long)
    minlength = num_classes**2
    bins = torch.bincount(unique_mapping, minlength=minlength)
    confmat = bins.reshape(num_classes, num_classes)
    return confmat
