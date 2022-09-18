import torch
from torch.nn.functional import pad

from ..processing import normalize
from .convert import power_to_db
from .helpers import sync_agg
from .spectral import istft, melspectrogram, stft


def onset_strength(y, sr, hop_length=1024, n_fft=2048, aggregate=torch.mean):
    S = torch.abs(melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, fmax=11025.0))
    S = power_to_db(S)

    onset_env = S[:, 1:] - S[:, :-1]
    onset_env = torch.maximum(torch.zeros(()), onset_env)
    onset_env = sync_agg(onset_env, [slice(None)], aggregate=aggregate, axis=0)

    pad_width = 1 + n_fft // (2 * hop_length)  # Counter-act framing effects. Shift the onsets by n_fft / hop_length

    onset_env = pad(onset_env, (int(pad_width), 0, 0, 0), mode="constant")
    onset_env = onset_env[:, : S.shape[1]]

    return onset_env.squeeze()


def fourier_tempo_frequencies(sr, win_length=1024, hop_length=1024, device="cpu"):
    # sr / hop_length gets the frame rate
    # multiplying by 60 turns frames / sec into frames / minute
    rate = sr * 60 / float(hop_length)
    return torch.linspace(0, float(rate) / 2, int(1 + win_length // 2), device=device)


def fourier_tempogram(
    y=None, sr=22050, onset_envelope=None, hop_length=1024, win_length=1024, center=True, window=torch.hann_window
):
    if onset_envelope is None:
        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)
    return stft(onset_envelope, n_fft=win_length, hop_length=1, center=center, window=window)


def plp(y, sr, hop_length=1024, win_length=1024, tempo_min=60, tempo_max=180):
    onset_envelope = onset_strength(
        y=y, sr=sr, hop_length=hop_length, aggregate=lambda *args, **kwargs: torch.median(*args, **kwargs).values
    )

    # Step 2: get the fourier tempogram
    max_win_len = min(len(onset_envelope), win_length)
    ftgram = fourier_tempogram(onset_envelope=onset_envelope, sr=sr, hop_length=hop_length, win_length=max_win_len)

    # Step 3: pin to the feasible tempo range
    tempo_frequencies = fourier_tempo_frequencies(sr=sr, hop_length=hop_length, win_length=max_win_len, device=y.device)

    if tempo_min is not None:
        ftgram[tempo_frequencies < tempo_min] = 0
    if tempo_max is not None:
        ftgram[tempo_frequencies > tempo_max] = 0

    # Step 3: Discard everything below the peak
    ftmag = torch.log1p(1e6 * torch.abs(ftgram))

    peak_values = ftmag.max(axis=0, keepdims=True).values
    ftgram[ftmag < peak_values] = 0

    # Normalize to keep only phase information
    absmaxabs = torch.abs(ftgram.abs().max(axis=0, keepdim=True).values)
    ftgram = ftgram / (torch.finfo(ftgram.dtype).tiny ** 0.5 + absmaxabs)

    # Step 5: invert the Fourier tempogram to get the pulse
    pulse = istft(ftgram, n_fft=max_win_len, hop_length=1, length=len(onset_envelope))

    # Step 6: retain only the positive part of the pulse cycle
    pulse = torch.clamp(pulse, torch.zeros((), device=pulse.device), pulse.max())

    # Return the normalized pulse
    return normalize(pulse)
