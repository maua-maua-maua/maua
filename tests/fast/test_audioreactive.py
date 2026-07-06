"""CPU-fast coverage of the audio-reactive signal/latent/MIR primitives.

These are the building blocks every audio-reactive patch composes, so a bug here is
silent-but-everywhere. They run on CPU over the 2s `short_audio` clip.
"""

import torch


def test_signal_ops():
    # import from the submodule directly: the package __init__ star-imports pull scipy's
    # `signal` into the package namespace, so `audioreactive.signal` is not our module.
    from maua.audiovisual.audioreactive.signal import gaussian_filter, normalize, percentile_clip, resample

    x = torch.linspace(-3, 5, 100)

    resampled = resample(x, 50)
    assert resampled.shape[0] == 50, "resample must hit the requested length"

    n = normalize(x)
    assert torch.isclose(n.min(), torch.tensor(0.0)) and torch.isclose(n.max(), torch.tensor(1.0))

    # percentile_clip works off local peaks, so it needs an oscillating signal
    osc = torch.sin(torch.linspace(0, 20, 200)).abs()
    clipped = percentile_clip(osc.clone(), 90).squeeze()
    assert clipped.max() <= 1.0 + 1e-5 and clipped.shape == osc.shape

    smoothed = gaussian_filter(x.clone(), sigma=3)
    assert smoothed.shape == x.shape
    # smoothing must reduce the total variation of a noisy signal
    noisy = torch.randn(200)
    tv = lambda s: s.diff().abs().sum()
    assert tv(gaussian_filter(noisy, sigma=4)) < tv(noisy)


def test_latent_weighting_and_loops():
    from maua.audiovisual.audioreactive import latent

    low = torch.zeros(4, 8)
    high = torch.ones(4, 8)
    envelope = torch.linspace(0, 1, 20)

    seq = latent.single_weighted(low, high, envelope)
    assert seq.shape == (20, 4, 8)
    # endpoints must equal the pure latents they interpolate between
    assert torch.allclose(seq[0], low) and torch.allclose(seq[-1], high)

    palette = torch.randn(3, 4, 8)
    loop = latent.spline_loops(palette, size=30, n_loops=1)
    assert loop.shape == (30, 4, 8)
    # a spline loop should return near its starting point
    assert torch.allclose(loop[0], loop[-1], atol=0.2)


def test_mir_features(short_audio):
    import numpy as np

    from maua.audiovisual.audioreactive.audio import load_audio
    from maua.audiovisual.audioreactive.mir import chroma, volume

    audio, sr, _ = load_audio(str(short_audio))
    audio = audio.numpy()  # the mir helpers wrap librosa, which wants a numpy buffer

    vol = volume(audio, sr)
    assert vol.ndim == 1 and len(vol) > 1 and torch.isfinite(vol).all()

    chr = np.asarray(chroma(audio, sr))
    assert chr.shape[-1] == 12, "chroma has 12 pitch classes"
    assert np.isfinite(chr).all()
