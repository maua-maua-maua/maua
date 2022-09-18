import numpy as np
import torch
from torch.nn.functional import pad

from .convert import hz_to_octs
from .spectral import spectrogram


def estimate_tuning(y, sr, n_fft=2048, resolution=0.01, bins_per_octave=12, **kwargs):
    pitch, mag = piptrack(y=y, sr=sr, n_fft=n_fft, **kwargs)

    # Only count magnitude where frequency is > 0
    pitch_mask = pitch > 0

    if pitch_mask.any():
        threshold = torch.median(mag[pitch_mask])
    else:
        threshold = 0.0

    return pitch_tuning(
        pitch[(mag >= threshold) & pitch_mask],
        resolution=resolution,
        bins_per_octave=bins_per_octave,
    )


def piptrack(
    y,
    sr,
    n_fft=2048,
    hop_length=None,
    fmin=150.0,
    fmax=4000.0,
    threshold=0.1,
    window=torch.hann_window,
    center=True,
    pad_mode="reflect",
):
    S = spectrogram(y, n_fft=n_fft, hop_length=hop_length, window=window, center=center, pad_mode=pad_mode)

    # Make sure we're dealing with magnitudes
    S = torch.abs(S)

    # Truncate to feasible region
    fmin = max(fmin, 0)
    fmax = min(fmax, float(sr) / 2)

    fft_freqs = torch.linspace(0, float(sr) / 2, int(1 + n_fft // 2), device=y.device)

    # Do the parabolic interpolation everywhere,
    # then figure out where the peaks are
    # then restrict to the feasible range (fmin:fmax)
    avg = 0.5 * (S[2:] - S[:-2])

    shift = 2 * S[1:-1] - S[2:] - S[:-2]

    # Suppress divide-by-zeros.
    # Points where shift == 0 will never be selected by localmax anyway
    shift = avg / (shift + (torch.abs(shift) < torch.finfo(shift.dtype).tiny))

    # Pad back up to the same shape as S
    avg = pad(avg, ([0, 0, 1, 1]), mode="constant")
    shift = pad(shift, ([0, 0, 1, 1]), mode="constant")

    dskew = 0.5 * avg * shift

    # Pre-allocate output
    pitches = torch.zeros_like(S)
    mags = torch.zeros_like(S)

    # Clip to the viable frequency range
    freq_mask = ((fmin <= fft_freqs) & (fft_freqs < fmax)).reshape((-1, 1))

    ref_value = threshold * torch.max(S, axis=0).values

    idx = torch.argwhere(freq_mask & localmax(S * (S > ref_value)))

    # Store pitch and magnitude
    pitches[idx[:, 0], idx[:, 1]] = (idx[:, 0] + shift[idx[:, 0], idx[:, 1]]) * float(sr) / n_fft

    mags[idx[:, 0], idx[:, 1]] = S[idx[:, 0], idx[:, 1]] + dskew[idx[:, 0], idx[:, 1]]

    return pitches, mags


def localmax(x):
    x_pad = pad(x, (0, 0, 1, 1))

    inds1 = [slice(None)] * x.ndim
    inds1[0] = slice(0, -2)

    inds2 = [slice(None)] * x.ndim
    inds2[0] = slice(2, x_pad.shape[0])

    return (x > x_pad[tuple(inds1)]) & (x >= x_pad[tuple(inds2)])


def pitch_tuning(frequencies, resolution=0.01, bins_per_octave=12):
    frequencies = torch.atleast_1d(frequencies)

    # Trim out any DC components
    frequencies = frequencies[frequencies > 0]

    if not torch.any(frequencies):
        return 0.0

    # Compute the residual relative to the number of bins
    residual = (bins_per_octave * hz_to_octs(frequencies)) % 1.0

    # Are we on the wrong side of the semitone?
    # A residual of 0.95 is more likely to be a deviation of -0.05
    # from the next tone up.
    residual[residual >= 0.5] -= 1.0

    bins = int(np.ceil(1.0 / resolution))
    counts = torch.histc(residual, bins=bins, min=-0.5, max=0.5)
    tuning = torch.linspace(-0.5, 0.5, bins + 1, device=frequencies.device)

    # return the histogram peak
    return tuning[torch.argmax(counts)]
