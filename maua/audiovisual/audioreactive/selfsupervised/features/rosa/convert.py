import librosa as rosa
import numpy as np
import torch
from torch.nn.functional import conv2d


def power_to_db(magnitude, ref_value=torch.ones(()), amin=torch.tensor(1e-10), top_db=80.0):
    log_spec = 10.0 * torch.log10(torch.maximum(amin, magnitude))
    log_spec -= 10.0 * torch.log10(torch.maximum(amin, ref_value))
    if top_db is not None:
        log_spec = torch.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def hz_to_mel(frequencies, htk=False, device="cpu"):
    frequencies = torch.tensor(frequencies, device=device)

    if htk:
        return 2595.0 * torch.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + torch.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + torch.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * torch.exp(logstep * (mels - min_log_mel))

    return freqs


def cq_to_chroma(
    n_input,
    device,
    bins_per_octave=12,
    n_chroma=12,
    fmin=None,
    window=None,
    base_c=True,
    dtype=torch.float,
):
    # How many fractional bins are we merging?
    n_merge = float(bins_per_octave) / n_chroma

    if fmin is None:
        fmin = note_to_hz("C1")

    # Tile the identity to merge fractional bins
    cq_to_ch = torch.repeat_interleave(torch.eye(n_chroma, device=device), round(n_merge), dim=1)

    # Roll it left to center on the target bin
    cq_to_ch = torch.roll(cq_to_ch, -int(n_merge // 2), dims=1)

    # How many octaves are we repeating?
    n_octaves = np.ceil(float(n_input) / bins_per_octave)

    # Repeat and trim
    cq_to_ch = torch.tile(cq_to_ch, (1, int(n_octaves)))[:, :n_input]

    # What's the note number of the first bin in the CQT? midi uses 12 bins per octave here
    midi_0 = hz_to_midi(fmin) % 12

    if base_c:
        # rotate to C
        roll = midi_0
    else:
        # rotate to A
        roll = midi_0 - 9

    # Adjust the roll in terms of how many chroma we want out
    # We need to be careful with rounding here
    roll = int(torch.round(roll * (n_chroma / 12.0)))

    # Apply the roll
    cq_to_ch = torch.roll(cq_to_ch, roll, dims=0).to(dtype)

    if window is not None:
        cq_to_ch = conv2d(cq_to_ch, window, padding="same")

    return cq_to_ch


def hz_to_octs(frequencies, tuning=0.0, bins_per_octave=12):
    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)
    return torch.log2(frequencies / (float(A440) / 16))


def hz_to_midi(frequencies):
    return 12 * (np.log2(frequencies) - np.log2(440.0)) + 69


def note_to_hz(note):
    return torch.tensor(rosa.core.convert.note_to_hz(note)).float()
