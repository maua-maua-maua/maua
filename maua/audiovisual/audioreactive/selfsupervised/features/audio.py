from functools import partial

import librosa as rosa
import torch
from torch.nn.functional import pad

from .processing import emphasize, gaussian_filter, normalize
from .rosa.beat import onset_strength, plp
from .rosa.convert import power_to_db
from .rosa.spectral import chroma_cens, chroma_cqt, dct, hpss, istft, melspectrogram, spectrogram, stft


def harmonic(audio, margin=8.0):
    y_stft = stft(audio)
    stft_harm = hpss(y_stft, margin=margin)[0]
    y_harm = istft(stft_harm, length=len(audio))
    return y_harm


def percussive(audio, margin=8.0):
    y_stft = stft(audio)
    stft_perc = hpss(y_stft, margin=margin)[1]
    y_perc = istft(stft_perc, length=len(audio))
    return y_perc


def onsets(audio, sr):
    return normalize(onset_strength(percussive(audio), sr).unsqueeze(-1))


def rms(y, sr, frame_length=2048, hop_length=1024, center=True, pad_mode="reflect"):
    if center:
        p = int(frame_length // 2)
        y = pad(y.unsqueeze(0), (p, p), mode=pad_mode).squeeze()
    x = y.unfold(0, frame_length, hop_length)[:-1]
    power = torch.mean(torch.abs(x) ** 2, dim=1)
    return torch.sqrt(power).unsqueeze(-1)


def drop_strength(audio, sr):
    return emphasize(gaussian_filter(rms(audio, sr), 10), strength=10, percentile=50).unsqueeze(1)


def chromagram(audio, sr):
    return chroma_cens(harmonic(audio), sr).T


def tonnetz(y, sr, chroma_fn=lambda a, sr: chromagram(a, sr).T):
    chroma = chroma_fn(y, sr)
    dim_map = torch.linspace(0, 12, chroma.shape[0], device=y.device)  # Generate Transformation matrix
    scale = torch.tensor([7.0 / 6, 7.0 / 6, 3.0 / 2, 3.0 / 2, 2.0 / 3, 2.0 / 3], device=y.device)
    V = scale.reshape(-1, 1) * dim_map
    V[::2] -= 0.5  # Even rows compute sin()
    R = torch.tensor([1, 1, 1, 1, 0.5, 0.5], device=y.device)  # Fifths  # Minor  # Major
    phi = R[:, None] * torch.cos(torch.pi * V)
    ton = phi @ (chroma / chroma.norm(p=1, dim=0))
    return ton.T


def mfcc(y, sr, n_mfcc=20, norm=False, **kwargs):
    S = power_to_db(melspectrogram(y, sr, **kwargs))
    M = dct(S.permute(1, 0), norm="ortho").permute(1, 0)[:n_mfcc]
    if norm == True:
        M = M / M.norm(p=2)
    return M.T


def pulse(audio, sr):
    return plp(percussive(audio), sr).unsqueeze(-1)


def spectral_contrast(
    y,
    sr,
    n_fft=2048,
    hop_length=1024,
    window=torch.hann_window,
    center=True,
    pad_mode="reflect",
    fmin=200.0,
    n_bands=6,
    quantile=0.02,
    linear=False,
):
    S = spectrogram(y=y, n_fft=n_fft, hop_length=hop_length, window=window, center=center, pad_mode=pad_mode)

    freq = torch.linspace(0, float(sr) / 2, int(1 + n_fft // 2), device=y.device)

    octa = torch.zeros(n_bands + 2, device=y.device)
    octa[1:] = fmin * (2.0 ** torch.arange(0, n_bands + 1, device=y.device))

    valley = torch.zeros((n_bands + 1, S.shape[1]), device=y.device)
    peak = torch.zeros_like(valley)

    for k, (f_low, f_high) in enumerate(zip(octa[:-1], octa[1:])):
        current_band = torch.logical_and(freq >= f_low, freq <= f_high)

        idx = current_band.flatten().nonzero()

        if k > 0:
            current_band[idx[0] - 1] = True

        if k == n_bands:
            current_band[idx[-1] + 1 :] = True

        sub_band = S[current_band]

        if k < n_bands:
            sub_band = sub_band[:-1]

        # Always take at least one bin from each side
        idx = torch.round(quantile * torch.sum(current_band))
        idx = int(torch.maximum(idx, torch.ones((), device=y.device)))

        sortedr = torch.sort(sub_band, dim=0).values

        valley[k] = torch.mean(sortedr[:idx], dim=0)
        peak[k] = torch.mean(sortedr[-idx:], dim=0)

    if linear:
        return (peak - valley).T
    else:
        return (power_to_db(peak) - power_to_db(valley)).T


def spectral_flatness(
    y, sr, n_fft=2048, hop_length=1024, window=torch.hann_window, center=True, pad_mode="reflect", amin=1e-10, power=2.0
):
    S = spectrogram(y=y, n_fft=n_fft, hop_length=hop_length, power=1.0, window=window, center=center, pad_mode=pad_mode)
    S_thresh = torch.maximum(torch.tensor(amin, device=y.device), S**power)
    gmean = torch.exp(torch.mean(torch.log(S_thresh), axis=0))
    amean = torch.mean(S_thresh, axis=0)
    return (gmean / amean).unsqueeze(-1)


if __name__ == "__main__":
    from time import time

    import librosa.display
    import matplotlib.pyplot as plt
    import torchaudio as ta

    audio, sr = ta.load("/home/hans/datasets/wavefunk/Ouroboromorphism_49_109.flac")
    SR = 1024 * 24
    audio, sr = ta.functional.resample(audio, sr, SR), SR
    audio = audio.mean(0).cuda()
    audio.requires_grad_(True)

    # fmt:off
    features = [
        ("onsets", partial(rosa.onset.onset_strength, hop_length=1024), onset_strength, onsets, "plot"),
        ("rms", partial(rosa.feature.rms, hop_length=1024), rms, drop_strength, "plot"),
        ("chroma", partial(rosa.feature.chroma_cens, hop_length=1024), chroma_cens, lambda a,sr: chromagram(a, sr).T, "chroma"),
        ("tonnetz", partial(rosa.feature.tonnetz, hop_length=1024), lambda a, sr : tonnetz(a, sr, chroma_fn=chroma_cqt).T, lambda a, sr : tonnetz(a, sr).T, "tonnetz"),
        ("mfcc", partial(rosa.feature.mfcc, hop_length=1024), lambda a,sr: mfcc(a,sr).T, lambda a,sr: mfcc(a,sr).T, None),
        ("pulse", partial(rosa.beat.plp, win_length=1024, tempo_min=60, tempo_max=180, hop_length=1024), plp, pulse, "plot"),
        ("contrast", partial(rosa.feature.spectral_contrast, hop_length=1024), lambda a,sr: spectral_contrast(a,sr).T, lambda a,sr: spectral_contrast(a,sr).T, None),
        ("flatness", lambda y, s: rosa.feature.spectral_flatness(y, hop_length=1024), spectral_flatness, spectral_flatness, 'plot'),
    ]
    # fmt:on

    trials = 10

    fig, ax = plt.subplots(3, len(features), figsize=(8 * len(features), 18))
    for f, (name, rosa_fn, torch_fn, noice_fn, y_axis) in enumerate(features):
        audio.grad = None

        a_np = audio.squeeze().detach().cpu().numpy()
        t = time()
        for _ in range(trials):
            f_np = rosa_fn(a_np, sr)
        t_np = (time() - t) / trials
        f_np = torch.from_numpy(f_np).float().cuda()

        t = time()
        for _ in range(trials):
            f_th = torch_fn(audio, sr)
            torch.cuda.synchronize()
        t_th = (time() - t) / trials

        diff = torch.abs(f_np - f_th)
        diff.sum().backward()

        print(
            f"{f'{name} numpy'.ljust(20)} {f_np.min().item():.4f}, {f_np.mean().item():.4f}, {f_np.max().item():.4f}, {tuple(f_np.shape)}, {t_np*1000:.4f} ms"
        )
        print(
            f"{f'{name} torch'.ljust(20)} {f_th.min().item():.4f}, {f_th.mean().item():.4f}, {f_th.max().item():.4f}, {tuple(f_th.shape)}, {t_th*1000:.4f} ms"
        )
        print(f"{name} diff".ljust(20), diff.min().item(), diff.mean().item(), diff.max().item())
        print(f"{name} grad norm".ljust(20), torch.norm(audio.grad).item(), "\n")

        f_np = f_np.squeeze().detach().cpu().numpy()
        f_th = f_th.squeeze().detach().cpu().numpy()
        f_noice = noice_fn(audio, sr).squeeze().detach().cpu().numpy()

        if y_axis == "plot":
            ax[0, f].plot(f_np)
            ax[1, f].plot(f_th)
            ax[2, f].plot(f_noice)
        else:
            librosa.display.specshow(f_np, y_axis=y_axis, x_axis="time", ax=ax[0, f])
            librosa.display.specshow(f_th, y_axis=y_axis, x_axis="time", ax=ax[1, f])
            librosa.display.specshow(f_noice, y_axis=y_axis, x_axis="time", ax=ax[2, f])

        ax[0, f].set_title(name)
    ax[0, 0].set_ylabel("numpy")
    ax[1, 0].set_ylabel("torch")
    ax[2, 0].set_ylabel("noice")
    plt.savefig("output/rosa_torch_audio_feature_comparison.pdf")
    plt.close()
