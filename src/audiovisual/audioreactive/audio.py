import os
from pathlib import Path

import joblib
import librosa as rosa
import torch
import torchaudio
from openunmix.predict import separate
from scipy import signal
from torchaudio.functional import resample

from . import cache_to_workspace


def load_audio(audio_file, offset=0, duration=-1, cache=True):
    """Handles loading of audio files. Automatically caches to .npy files to increase loading speed.

    Args:
        audio_file (str): Path to audio file to load
        offset (float, optional): Time (in seconds) to start from. Defaults to 0.
        duration (float, optional): Length of time to load in. Defaults to -1 (full duration).
        cache (bool): Whether to cache the raw audio file or not

    Returns:
        audio   : audio signal
        sr      : sample rate of audio
        duration: duration of audio in seconds
    """
    audio_dur = rosa.get_duration(filename=audio_file)
    if duration == -1 or audio_dur < duration:
        duration = audio_dur
        if offset != 0:
            duration -= offset

    cache_file = (
        f"workspace/audio_cache/{Path(audio_file.replace('/','_')).stem}"
        + ("" if duration == -1 else f"_length{duration}")
        + ("" if offset == 0 else f"_start{offset}")
        + ".npy"
    )
    if cache and not os.path.exists(cache_file):
        audio, sr = torchaudio.load(audio_file)
        audio = audio[:, int(offset * sr) : int((offset + duration) * sr)].mean(0).numpy()
        joblib.dump((audio, sr), cache_file)
    else:
        audio, sr = joblib.load(cache_file)

    return torch.from_numpy(audio), sr, duration


@cache_to_workspace("unmixed")
def unmix(audio, sr):
    a = resample(torch.from_numpy(audio).add(1).div(2), sr, 44100).mul(2).add(-1).clamp(-1, 1)
    vocals, drums, bass, instruments = separate(a, rate=44100, niter=3, device="cpu").values()

    vocals = resample(vocals, 44100, sr).squeeze().mean(0).cpu().numpy()
    drums = resample(drums, 44100, sr).squeeze().mean(0).cpu().numpy()
    bass = resample(bass, 44100, sr).squeeze().mean(0).cpu().numpy()
    instruments = resample(instruments, 44100, sr).squeeze().mean(0).cpu().numpy()

    return vocals, drums, bass, instruments


def unmixed(audio, sr, stem="all"):
    vocals, drums, bass, instruments = unmix(audio, sr)

    if stem == "vocals":
        return vocals
    if stem == "drums":
        return drums
    if stem == "bass":
        return bass
    if stem == "instruments":
        return instruments

    return vocals, drums, bass, instruments


@cache_to_workspace("spleeted")
def spleeted(audio, sr):
    raise NotImplementedError()


@cache_to_workspace("harmonic")
def harmonic(audio, sr, margin=8):
    y = rosa.effects.harmonic(y=audio, margin=margin)
    return y


@cache_to_workspace("percussive")
def percussive(audio, sr, margin=8):
    y = rosa.effects.percussive(y=audio, margin=margin)
    return y


@cache_to_workspace("low_passed")
def low_pass(audio, sr, fmax=200, db_per_octave=12):
    y = signal.sosfilt(signal.butter(db_per_octave, fmax, "low", fs=sr, output="sos"), audio)
    return y


@cache_to_workspace("high_passed")
def high_pass(audio, sr, fmin=3000, db_per_octave=12):
    y = signal.sosfilt(signal.butter(db_per_octave, fmin, "high", fs=sr, output="sos"), audio)
    return y


@cache_to_workspace("band_passed")
def band_pass(audio, sr, fmin=200, fmax=3000, db_per_octave=12):
    y = signal.sosfilt(signal.butter(db_per_octave, [fmin, fmax], "band", fs=sr, output="sos"), audio)
    return y
