import numpy as np
import torch
from torch.nn.functional import conv1d
from torchcubicspline import natural_cubic_spline_coeffs

from ..processing import median_filter2d
from .convert import cq_to_chroma, hz_to_mel, mel_to_hz


def stft(
    y, n_fft=2048, hop_length=1024, center=True, window=torch.hann_window, pad_mode="reflect", return_complex=True
):
    return torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        center=center,
        window=window(n_fft, device=y.device) if window is not None else None,
        pad_mode=pad_mode,
        return_complex=return_complex,
    )


def istft(spec, n_fft=2048, hop_length=1024, center=True, window=torch.hann_window, length=None):
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        center=center,
        window=window(n_fft, device=spec.device) if window is not None else None,
        length=length,
    )


def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def spectrogram(y, n_fft=2048, hop_length=1024, power=1, window=torch.hann_window, center=True, pad_mode="reflect"):
    y_stft = stft(y, n_fft=n_fft, hop_length=hop_length, center=center, window=window, pad_mode=pad_mode)[:, :-1]
    S = torch.abs(y_stft) ** power
    return S


def melspectrogram(
    y, sr, n_fft=2048, hop_length=1024, window=torch.hann_window, center=True, pad_mode="reflect", power=2.0, fmax=None
):
    S = spectrogram(y, n_fft=n_fft, hop_length=hop_length, power=power, window=window, center=center, pad_mode=pad_mode)
    mel_basis = mel(sr, n_fft, fmax=fmax, device=S.device)
    return mel_basis @ S


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False, device="cpu"):
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk, device=device)
    max_mel = hz_to_mel(fmax, htk=htk, device=device)
    mels = torch.linspace(min_mel, max_mel, n_mels, device=device)
    return mel_to_hz(mels, htk=htk)


def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, dtype=torch.float, device="cpu"):
    if fmax is None:
        fmax = float(sr) / 2

    n_mels = int(n_mels)
    weights = torch.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype, device=device)

    # Center freqs of each FFT bin
    fftfreqs = torch.linspace(0, float(sr) / 2, int(1 + n_fft // 2), device=device)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk, device=device)

    fdiff = torch.diff(mel_f)

    ramps = mel_f.reshape(-1, 1) - fftfreqs

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = torch.maximum(torch.zeros(()), torch.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]

    return weights


def magphase(D, power=1.0):
    mag = torch.abs(D)
    mag = mag**power
    phase = torch.exp(1.0j * torch.angle(D))
    return mag, phase


def softmask(X, X_ref, power=torch.ones(()), split_zeros=False):
    # Re-scale the input arrays relative to the larger value
    dtype = torch.float
    Z = torch.maximum(X, X_ref).to(dtype)
    bad_idx = Z < torch.finfo(dtype).tiny
    Z[bad_idx] = 1

    # For finite power, compute the softmask
    if np.isfinite(power):
        mask = (X / Z) ** power
        ref_mask = (X_ref / Z) ** power
        good_idx = ~bad_idx
        mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
        # Wherever energy is below energy in both inputs, split the mask
        if split_zeros:
            mask[bad_idx] = 0.5
        else:
            mask[bad_idx] = 0.0
    else:
        # Otherwise, compute the hard mask
        mask = X > X_ref

    return mask


def hpss(S, ks=31, power=2.0, margin=1.0):
    if torch.is_complex(S):
        S, phase = magphase(S)
    else:
        phase = 1.0

    # Compute median filters. Pre-allocation here preserves memory layout.
    harm = torch.empty_like(S)
    harm[:] = median_filter2d(S[None, None], k=(1, ks), p=(ks // 2, ks // 2, 0, 0)).squeeze()

    perc = torch.empty_like(S)
    perc[:] = median_filter2d(S[None, None], k=(ks, 1), p=(0, 0, ks // 2, ks // 2)).squeeze()

    split_zeros = margin == 1 and margin == 1
    mask_harm = softmask(harm, perc * margin, power=power, split_zeros=split_zeros)
    mask_perc = softmask(perc, harm * margin, power=power, split_zeros=split_zeros)
    return ((S * mask_harm) * phase, (S * mask_perc) * phase)


with torch.no_grad():
    QUANT_STEPS = [0.4, 0.2, 0.1, 0.05]  # from librosa.feature.spectral.chroma_cens
    Q_STEP = 0.25
    QUANT_WEIGHTS = [Q_STEP, Q_STEP, Q_STEP, Q_STEP]

    p1, p2, p3, p4 = np.diff(list(reversed(QUANT_STEPS + [0])))
    xs = [
        torch.linspace(-0.1, 0.025, 101)[:-1],
        torch.linspace(0.025, p1, 11)[:-1],
        torch.linspace(p1, p1 + p2, 11)[:-1],
        torch.linspace(p1 + p2, p1 + p2 + p3, 11)[:-1],
        torch.linspace(p1 + p2 + p3, 0.5, 11)[:-1],
        torch.linspace(0.5, 1.1, 100),
    ]
    ys = torch.cat(
        (
            0.5 * torch.ones(len(xs[0])),
            xs[1] / p1,
            (xs[2] - p1) / p2 + 1,
            (xs[3] - p1 - p2) / p3 + 2,
            (xs[4] - p1 - p2 - p3) / p4 + 3,
            4.5 * torch.ones(len(xs[5])),
        )
    )
    xs = torch.cat(xs)
    COEFFS = natural_cubic_spline_coeffs(xs, ys.reshape(1, -1, 1))  # pre-calculate spline coefficients


def spline_eval(t, coeffs):
    y, a, b, c, d = (c.to(t.device) for c in coeffs)
    maxlen = b.size(-2) - 1
    index = torch.bucketize(t.detach(), y) - 1
    index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]
    # this is fine. we'll never access the last element of y. this is correct behaviour
    fractional_part = t - y[index]
    fractional_part = fractional_part.unsqueeze(-1)
    inner = c[..., index, :] + d[..., index, :] * fractional_part
    inner = b[..., index, :] + inner * fractional_part
    return a[..., index, :] + inner * fractional_part


def r(w):
    return (w - 0.5) - torch.floor(w - 0.5) - 0.5


def m(alpha):
    return 1 / (1 + np.exp(-alpha)) - 0.5


def step_function(w, h=Q_STEP, alpha=20):  # alpha controls smoothness of step transition
    return h * (torch.floor(w - 0.5) + 1 / (2 * m(alpha)) * 1 / (1 + torch.exp(-2 * alpha * r(w))))


def spline_quantize(chroma):
    spline_mapped = torch.stack([spline_eval(c, COEFFS).squeeze() for c in chroma])
    return step_function(spline_mapped)


def test_spline_quantization():
    import matplotlib.pyplot as plt

    xt = torch.linspace(-1, 2, 1000)
    yt = spline_eval(xt).squeeze()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(xs, ys, color="tab:blue", alpha=0.5, label="original")
    ax[0].plot(xt, yt, color="tab:orange", alpha=0.5, label="spline approx")
    ax[0].set_xlim(-0.1, 1.1)
    ax[0].legend()
    ax[1].plot(xt, step_function(yt))
    ax[1].set_xlim(-0.1, 0.6)
    plt.tight_layout()
    plt.show()


def chroma_cens(
    y,
    sr,
    hop_length=1024,
    fmin=None,
    tuning=None,
    n_chroma=12,
    n_octaves=7,
    bins_per_octave=36,
    window=None,
    win_len_smooth=41,
    smoothing_window=torch.hann_window,
):
    chroma = chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        n_chroma=n_chroma,
        n_octaves=n_octaves,
        window=window,
        norm=False,
    )
    chroma = chroma / torch.norm(chroma, p=1, dim=0)  # L1-Normalization

    # - 0.05     --> 0
    # 0.05 - 0.1 --> 0.25
    # 0.1 - 0.2  --> 0.5
    # 0.2 - 0.4  --> 0.75
    # 0.4 +      --> 1.0
    chroma_quant = spline_quantize(chroma)

    if win_len_smooth:  # Apply temporal smoothing
        win = smoothing_window(win_len_smooth + 2, device=chroma_quant.device)
        win /= torch.sum(win)
        cens = conv1d(chroma_quant.unsqueeze(0), win.tile(12, 1, 1), groups=chroma.shape[0], padding="same").squeeze(0)
    else:
        cens = chroma_quant

    return cens / torch.norm(cens, p=2, dim=0)  # L2-Normalization


from .constantq import cqt


def chroma_cqt(
    y,
    sr,
    hop_length=1024,
    fmin=None,
    threshold=0.0,
    tuning=None,
    n_chroma=12,
    n_octaves=7,
    window=None,
    bins_per_octave=36,
    norm=True,
):

    # Build the CQT if we don't have one already
    C = torch.abs(
        cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_octaves * bins_per_octave,
            bins_per_octave=bins_per_octave,
            tuning=tuning,
        )
    )

    # Map to chroma
    cq_to_chr = cq_to_chroma(
        C.shape[0], C.device, bins_per_octave=bins_per_octave, n_chroma=n_chroma, fmin=fmin, window=window
    )
    chroma = cq_to_chr @ C

    if threshold is not None:
        chroma[chroma < threshold] = 0.0

    if norm:
        chroma = chroma / chroma.max()

    return chroma
