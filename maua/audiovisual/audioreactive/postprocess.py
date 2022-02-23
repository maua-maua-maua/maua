import torch
import torch.nn.functional as F


def normalize(x):
    """Normalize signal between 0 and 1

    Args:
        signal (np.array/torch.tensor): Signal to normalize

    Returns:
        np.array/torch.tensor: Normalized signal
    """
    y = x - x.min()
    y = y / y.max()
    return y


def percentile(signal, p):
    """Calculate percentile of signal

    Args:
        signal (np.array/torch.tensor): Signal to normalize
        p (int): [0-100]. Percentile to find

    Returns:
        int: Percentile signal value
    """
    k = 1 + round(0.01 * float(p) * (signal.numel() - 1))
    return signal.view(-1).kthvalue(k).values.item()


def percentile_clip(signal, percent):
    """Normalize signal between 0 and 1, clipping peak values above given percentile

    Args:
        signal (torch.tensor): Signal to normalize
        percent (int): [0-100]. Percentile to clip to

    Returns:
        torch.tensor: Clipped signal
    """
    result = []
    if len(signal.shape) < 2:
        signal = signal.unsqueeze(1)
    for sig in signal.unbind(1):
        locs = torch.arange(0, sig.shape[0])
        peaks = torch.ones(sig.shape, dtype=bool)
        main = sig.take(locs)

        plus = sig.take((locs + 1).clamp(0, sig.shape[0] - 1))
        minus = sig.take((locs - 1).clamp(0, sig.shape[0] - 1))
        peaks &= torch.gt(main, plus)
        peaks &= torch.gt(main, minus)

        sig = sig.clamp(0, percentile(sig[peaks], percent))
        sig /= sig.max()
        result.append(sig)
    return torch.stack(result, dim=1)


def compress(signal, threshold, ratio, invert=False):
    """Expand or compress signal. Values above/below (depending on invert) threshold are multiplied by ratio.

    Args:
        signal (torch.tensor): Signal to normalize
        threshold (int): Signal value above/below which to change signal
        ratio (float): Value to multiply signal with
        invert (bool, optional): Specifies if values above or below threshold are affected. Defaults to False.

    Returns:
        torch.tensor: Compressed/expanded signal
    """
    if invert:
        signal[signal < threshold] *= ratio
    else:
        signal[signal > threshold] *= ratio
    return normalize(signal)


def expand(signal, threshold, ratio, invert=False):
    """Alias for compress. Whether compression or expansion occurs is determined by values of threshold and ratio"""
    return compress(signal, threshold, ratio, invert)


def gaussian_filter(x, sigma, causal=None, mode="circular"):
    """Smooth tensors along time (first) axis with gaussian kernel.

    Args:
        x (torch.tensor): Tensor to be smoothed
        sigma (float): Standard deviation for gaussian kernel (higher value gives smoother result)
        causal (float, optional): Factor to multiply right side of gaussian kernel with. Lower value decreases effect of "future" values. Defaults to None.

    Returns:
        torch.tensor: Smoothed tensor
    """
    dim = len(x.shape)
    n_frames = x.shape[0]
    while len(x.shape) < 3:
        x = x[:, None]

    radius = min(int(sigma * 4), 3 * len(x))
    channels = x.shape[1]

    kernel = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-0.5 / sigma**2 * kernel**2)
    if causal is not None:
        kernel[radius + 1 :] *= 0 if not isinstance(causal, float) else causal
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)

    if radius > n_frames:  # prevent padding errors on short sequences
        x = F.pad(x, (n_frames, n_frames), mode=mode)
        print(
            f"WARNING: Gaussian filter radius ({int(sigma * 4)}) is larger than number of frames ({n_frames}).\n\t Filter size has been lowered to ({radius}). You might want to consider lowering sigma ({sigma})."
        )
        x = F.pad(x, (radius - n_frames, radius - n_frames), mode="replicate")
    else:
        x = F.pad(x, (radius, radius), mode=mode)

    x = F.conv1d(x, weight=kernel, groups=channels)

    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t, c, h, w)

    if len(x.shape) > dim:
        x = x.squeeze()

    return x
