import numpy as np
import torch
from torch.nn.functional import pad
from torchaudio.functional import resample

from .convert import note_to_hz
from .pitch import estimate_tuning
from .spectral import stft

HANN_BANDWIDTH = 1.50018310546875


def cqt(y, sr, hop_length=1024, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0, filter_scale=1, sparsity=0.01):
    # CQT is the special case of VQT with gamma=0
    return vqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        gamma=0,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=filter_scale,
        sparsity=sparsity,
    )


def vqt(
    y,
    sr,
    hop_length=1024,
    fmin=None,
    n_bins=84,
    gamma=None,
    bins_per_octave=12,
    tuning=0.0,
    filter_scale=1,
    sparsity=0.01,
):
    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    # Relative difference in frequency between any two consecutive bands
    alpha = 2.0 ** (1.0 / bins_per_octave) - 1

    if fmin is None:
        fmin = note_to_hz("C1").to(y)  # C1 by default

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)

    if gamma is None:
        gamma = 24.7 * alpha / 0.108

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # First thing, get the freqs of the top octave
    freqs = cqt_frequencies(n_bins, fmin, bins_per_octave=bins_per_octave)[-bins_per_octave:]

    fmin_t = torch.min(freqs)

    vqt_resp = []

    # Make sure our hop is long enough to support the bottom octave
    num_twos = __num_two_factors(hop_length)
    if num_twos < n_octaves - 1:
        raise Exception(
            "hop_length must be a positive integer multiple of 2^{0:d} for {1:d}-octave CQT/VQT".format(
                n_octaves - 1, n_octaves
            )
        )

    # Now do the recursive bit
    my_y, my_sr, my_hop = y, sr, hop_length

    # Iterate down the octaves
    for i in range(n_octaves):
        # Resample (except first time
        if i > 0:
            my_y = resample(my_y, my_sr, my_sr / 2, resampling_method="kaiser_window")
            my_y *= np.sqrt(2)  # rescale signal to keep approximately equal total energy
            my_sr /= 2.0
            my_hop //= 2

        fft_basis, n_fft, _ = __cqt_filter_fft(
            sr=my_sr,
            fmin=fmin_t * 2.0**-i,
            n_bins=n_filters,
            bins_per_octave=bins_per_octave,
            filter_scale=filter_scale,
            sparsity=sparsity,
            gamma=gamma,
        )
        # Re-scale the filters to compensate for downsampling
        fft_basis *= np.sqrt(2**i)

        # Compute the vqt filter response and append to the stack
        vqt_resp.append(__cqt_response(my_y, n_fft, my_hop, fft_basis))

    V = __trim_stack(vqt_resp, n_bins)

    lengths = constant_q_lengths(
        sr,
        fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        filter_scale=filter_scale,
        gamma=gamma,
    )
    V /= torch.sqrt(lengths[:, None])

    return V


def __cqt_filter_fft(sr, fmin, n_bins, bins_per_octave, filter_scale, sparsity, hop_length=None, gamma=0.0):
    basis, lengths = constant_q(
        sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        filter_scale=filter_scale,
        pad_fft=True,
        gamma=gamma,
    )

    # Filters are padded up to the nearest integral power of 2
    n_fft = basis.shape[1]

    if hop_length is not None and n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length))):
        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

    # re-normalize bases with respect to the FFT window length
    basis *= lengths[:, None] / float(n_fft)

    # FFT and retain only the non-negative frequencies
    fft_basis = torch.fft.fft(basis, n=n_fft, axis=1)[:, : (n_fft // 2) + 1]

    # sparsify the basis
    fft_basis = sparsify_rows(fft_basis, quantile=sparsity)

    return fft_basis, n_fft, lengths


def sparsify_rows(x, quantile=0.01):
    mags = torch.abs(x)
    norms = torch.sum(mags, axis=1, keepdims=True)

    mag_sort = torch.sort(mags, axis=1).values
    cumulative_mag = torch.cumsum(mag_sort / norms, axis=1)

    threshold_idx = torch.argmin((cumulative_mag < quantile).to(torch.uint8), axis=1)

    idxs, vals = [], []
    for i, j in enumerate(threshold_idx):
        idx = torch.where(mags[i] >= mag_sort[i, j])[0]
        idxs.append(torch.cartesian_prod(torch.tensor([i], device=idx.device), idx))
        vals.append(x[i, idx])

    return torch.sparse_coo_tensor(
        torch.cat(idxs).permute(1, 0), torch.cat(vals), size=x.shape, dtype=x.dtype, device=x.device
    )


def __trim_stack(cqt_resp, n_bins):
    """Helper function to trim and stack a collection of CQT responses"""

    max_col = min(c_i.shape[-1] for c_i in cqt_resp)
    cqt_out = torch.empty((n_bins, max_col), dtype=cqt_resp[0].dtype, device=cqt_resp[0].device)

    # Copy per-octave data into output array
    end = n_bins
    for c_i in cqt_resp:
        # By default, take the whole octave
        n_oct = c_i.shape[0]
        # If the whole octave is more than we can fit, take the highest bins from c_i
        if end < n_oct:
            cqt_out[:end] = c_i[-end:, :max_col]
        else:
            cqt_out[end - n_oct : end] = c_i[:, :max_col]

        end -= n_oct

    return cqt_out


def __cqt_response(y, n_fft, hop_length, fft_basis, mode="reflect"):
    D = stft(y, n_fft=n_fft, hop_length=hop_length, window=None, pad_mode=mode)[:, :-1]
    return fft_basis.mm(D)


def __num_two_factors(x):
    """Return how many times integer x can be evenly divided by 2.

    Returns 0 for non-positive integers.
    """
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2

    return num_twos


def cqt_frequencies(n_bins, fmin, bins_per_octave=12, tuning=0.0):
    correction = 2.0 ** (float(tuning) / bins_per_octave)
    frequencies = 2.0 ** (torch.arange(0, n_bins, dtype=torch.float, device=fmin.device) / bins_per_octave)
    return correction * fmin * frequencies


def constant_q_lengths(sr, fmin, n_bins=84, bins_per_octave=12, filter_scale=1, gamma=0):
    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    Q = float(filter_scale) / alpha
    freq = fmin * (2.0 ** (torch.arange(n_bins, dtype=torch.float, device=fmin.device) / bins_per_octave))
    lengths = Q * sr / (freq + gamma / alpha)  # Convert frequencies to filter lengths
    return lengths


def constant_q(
    sr,
    fmin=None,
    n_bins=84,
    bins_per_octave=12,
    filter_scale=1,
    pad_fft=True,
    gamma=0,
):
    if fmin is None:
        fmin = note_to_hz("C1")

    # Pass-through parameters to get the filter lengths
    lengths = constant_q_lengths(
        sr, fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, filter_scale=filter_scale, gamma=gamma
    )

    freqs = fmin * (2.0 ** (torch.arange(n_bins, dtype=torch.float, device=fmin.device) / bins_per_octave))

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Build the filter: note, length will be ceil(ilen)
        ilen2 = torch.div(ilen, 2, rounding_mode="floor")
        sig = torch.exp(
            torch.arange(-ilen2, ilen2, dtype=torch.float, device=freqs.device) * 1j * 2 * torch.pi * freq / sr
        )
        sig = sig * torch.hann_window(len(sig), device=sig.device)  # Apply the windowing function
        sig = sig / sig.norm(p=1, dim=0)  # Normalize
        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0 ** (torch.ceil(torch.log2(max_len))))
    else:
        max_len = int(torch.ceil(max_len))

    filters = torch.stack([pad_center(filt, max_len) for filt in filters])

    return filters, lengths


def pad_center(data, size, axis=-1):
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    return pad(data, (lpad, int(size - n - lpad)), mode="constant")
