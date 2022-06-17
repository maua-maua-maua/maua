import librosa as rosa
import madmom as mm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
import sklearn.cluster
import torch

from . import cache_to_workspace
from .audio import harmonic, percussive
from .signal import percentile_clip


@cache_to_workspace("onsets")
def onsets(audio, sr, type="mm", prepercussive=4):
    """Creates onset envelope from audio

    Args:
        audio (np.array): Audio signal
        sr (int): Sampling rate of the audio
        prepercussive (int, optional): For percussive source separation, higher values create more extreme separations. Defaults to 4.
        type (str, optional): ["rosa", "mm"]. Whether to use librosa or madmom for onset analysis. Madmom is slower but often more accurate. Defaults to "mm".

    Returns:
        torch.tensor, shape=(n_frames,): Onset envelope
    """
    if prepercussive:
        audio = percussive(audio, sr)

    if type == "rosa":
        onset = rosa.onset.onset_strength(y=audio, sr=sr)

    elif type == "mm":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=sr)
        sig_frames = mm.audio.signal.FramedSignal(sig, frame_size=2048, hop_length=512)
        stft = mm.audio.stft.ShortTimeFourierTransform(sig_frames, circular_shift=True)
        spec = mm.audio.spectrogram.Spectrogram(stft, circular_shift=True)
        filt_spec = mm.audio.spectrogram.FilteredSpectrogram(spec, num_bands=24)

        spectral_diff = mm.features.onsets.spectral_diff(filt_spec)
        spectral_flux = mm.features.onsets.spectral_flux(filt_spec)
        superflux = mm.features.onsets.superflux(filt_spec)
        complex_flux = mm.features.onsets.complex_flux(filt_spec)
        modified_kullback_leibler = mm.features.onsets.modified_kullback_leibler(filt_spec)

        onset = np.mean(
            [
                spectral_diff / spectral_diff.max(),
                spectral_flux / spectral_flux.max(),
                superflux / superflux.max(),
                complex_flux / complex_flux.max(),
                modified_kullback_leibler / modified_kullback_leibler.max(),
            ],
            axis=0,
        )

    onset = percentile_clip(torch.from_numpy(onset), 95).numpy()

    return torch.from_numpy(onset.squeeze().astype(np.float32))


@cache_to_workspace("volume")
def volume(audio, sr):
    """Creates RMS envelope from audio

    Args:
        audio (np.array): Audio signal

    Returns:
        torch.tensor, shape=(n_frames,): RMS envelope
    """
    vol = rosa.feature.rms(audio)
    vol -= vol.min()
    vol /= vol.max()
    return torch.from_numpy(vol.squeeze().astype(np.float32))


@cache_to_workspace("chroma")
def chroma(audio, sr, type="cens", nearest_neighbor=True, preharmonic=4, notes=12):
    """Creates chromagram for the harmonic component of the audio

    Args:
        audio (np.array): Audio signal
        sr (int): Sampling rate of the audio
        type (str, optional): ["cens", "cqt", "stft", "deep", "clp"]. Which strategy to use to calculate the chromagram. Defaults to "cens".
        nearest_neighbor (bool, optional): Whether to post process using nearest neighbor smoothing. Defaults to True.
        preharmonic (int, optional): For harmonic source separation, higher values create more extreme separations. Defaults to 4.
        notes (int, optional): Use only the most prominent notes (returns a smaller chromagram).

    Returns:
        torch.tensor, shape=(n_frames, 12): Chromagram
    """
    if preharmonic:
        audio = harmonic(audio, sr, preharmonic)

    if type == "cens":
        ch = np.rollaxis(rosa.feature.chroma_cens(y=audio, sr=sr), 1, 0)
    elif type == "cqt":
        ch = np.rollaxis(rosa.feature.chroma_cqt(y=audio, sr=sr), 1, 0)
    elif type == "stft":
        ch = np.rollaxis(rosa.feature.chroma_stft(y=audio, sr=sr), 1, 0)
    elif type == "deep":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=sr)
        ch = mm.audio.chroma.DeepChromaProcessor().process(sig)
    elif type == "clp":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=sr)
        ch = mm.audio.chroma.CLPChromaProcessor().process(sig)
    else:
        print("chroma type not recognized, options are: [cens, cqt, deep, clp, or stft]. defaulting to cens...")
        ch = np.rollaxis(rosa.feature.chroma_cens(y=audio, sr=sr), 1, 0)

    if nearest_neighbor:
        ch = np.minimum(ch, rosa.decompose.nn_filter(ch, aggregate=np.median, metric="cosine"))

    if notes < 12:
        ch = ch[:, np.argsort(-ch.sum(0))[:notes]]

    ch -= ch.min()
    ch /= ch.max() + 1e-8
    return ch.squeeze().astype(np.float32)


@cache_to_workspace("tonnetz")
def tonnetz(audio, sr, type="cens", nearest_neighbor=True, preharmonic=4):
    ch = chroma(audio, sr, type=type, nearest_neighbor=nearest_neighbor, preharmonic=preharmonic)
    fps = ch.shape[1] / (len(audio) / sr)
    ton = np.rollaxis(rosa.feature.tonnetz(chroma=np.rollaxis(ch, 1, 0), sr=fps), 1, 0)
    ton -= ton.min()
    ton /= ton.max()
    return torch.from_numpy(ton.squeeze().astype(np.float32))


@cache_to_workspace("pitch_track")
def pitch_track(audio, sr, preharmonic=4):
    if preharmonic:
        audio = harmonic(audio, sr, preharmonic)
    pitches, magnitudes = rosa.piptrack(y=audio, sr=sr)
    average_pitch = np.average(pitches, axis=0, weights=magnitudes + 1e-8)
    return torch.from_numpy(average_pitch.squeeze().astype(np.float32))


@cache_to_workspace("spectral_max")
def spectral_max(audio, sr, n_mels=512):
    spectrum = rosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    spectrum = np.amax(spectrum, axis=0)
    spectrum -= spectrum.min()
    spectrum /= spectrum.max()
    return torch.from_numpy(spectrum.squeeze().astype(np.float32))


@cache_to_workspace("pitch_dominance")
def pitch_dominance(audio, sr, type="cens", nearest_neighbor=True, preharmonic=4):
    chromagram = chroma(audio, sr, type=type, nearest_neighbor=nearest_neighbor, preharmonic=preharmonic)
    chromagram_norm = chromagram / chromagram.sum(axis=1, keepdims=1)
    chromagram_sum = np.sum(chromagram_norm, axis=0)
    pitches_sorted = np.argsort(chromagram_sum)[::-1]
    return torch.from_numpy(pitches_sorted)


@cache_to_workspace("pulse")
def pulse(audio, sr, prior="lognorm", type="mm", prepercussive=4):
    onset_env = onsets(audio, sr, type=type, prepercussive=prepercussive)
    fps = len(onset_env) / (len(audio) / sr)

    if prior == "uniform":
        prior = scipy.stats.uniform(30, 300)
    elif prior == "lognorm":
        prior = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
    else:
        prior = None

    pul = rosa.beat.plp(onset_envelope=onset_env, sr=fps, prior=prior)
    pul = rosa.util.normalize(pul)
    return torch.from_numpy(pul.squeeze().astype(np.float32))


def round_to_nearest_half(number):
    return round(number * 2) / 2


@cache_to_workspace("tempo")
def tempo(audio, sr, prior="uniform", type="mm", prepercussive=4):
    onset_env = onsets(audio, sr, type=type, prepercussive=prepercussive)

    if prior == "uniform":
        prior = scipy.stats.uniform(60, 200)
    elif prior == "lognorm":
        prior = scipy.stats.lognorm(loc=np.log(120), scale=60, s=1)
    else:
        prior = None

    ac_global = rosa.autocorrelate(onset_env, max_size=512)
    ac_global = rosa.util.normalize(ac_global)

    peaks = torch.topk(torch.from_numpy(ac_global), 10).indices.numpy()
    peaks = peaks[np.greater(peaks, 3)]
    peaks = peaks[np.less(peaks, len(ac_global))]
    tempos_ac = rosa.tempo_frequencies(512)[peaks]
    for t in range(len(tempos_ac)):
        while tempos_ac[t] < 80:
            tempos_ac[t] *= 2
        while tempos_ac[t] > 200:
            tempos_ac[t] /= 2

    tempo = rosa.beat.tempo(onset_envelope=onset_env, prior=prior).squeeze()

    return [round_to_nearest_half(bpm) for bpm in (tempo, *tempos_ac)]


@cache_to_workspace("segmentation")
def laplacian_segmentation(audio, sr, k=5, plot=False):
    """Segments the audio with pattern recurrence analysis
    From https://librosa.org/doc/latest/auto_examples/plot_segmentation.html#sphx-glr-auto-examples-plot-segmentation-py%22

    Args:
        audio (np.array): Audio signal
        sr (int): Sampling rate of the audio
        k (int, optional): Number of labels to use during segmentation. Defaults to 5.
        plot (bool, optional): Whether to show plot of found segmentation. Defaults to False.

    Returns:
        tuple(list, list): List of starting timestamps and labels of found segments
    """
    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = rosa.amplitude_to_db(
        np.abs(rosa.cqt(y=audio, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)),
        ref=np.max,
    )

    # make CQT beat-synchronous to reduce dimensionality
    tempo, beats = rosa.beat.beat_track(y=audio, sr=sr, trim=False)
    Csync = rosa.util.sync(C, beats, aggregate=np.median)

    # build a weighted recurrence matrix using beat-synchronous CQT
    R = rosa.segment.recurrence_matrix(Csync, width=3, mode="affinity", sym=True)
    # enhance diagonals with a median filter
    df = rosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))

    # build the sequence matrix using mfcc-similarity
    mfcc = rosa.feature.mfcc(y=audio, sr=sr)
    Msync = rosa.util.sync(mfcc, beats)
    path_distance = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    # compute the balanced combination
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec) ** 2)

    A = mu * Rf + (1 - mu) * R_path
    # compute the normalized laplacian and its spectral decomposition
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)
    # median filter to smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    # cumulative normalization for symmetric normalized laplacian eigenvectors
    Cnorm = np.cumsum(evecs**2, axis=1) ** 0.5

    X = evecs[:, :k] / Cnorm[:, k - 1 : k]

    # use first k components to cluster beats into segments
    seg_ids = sklearn.cluster.KMeans(n_clusters=k).fit_predict(X)

    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])  # locate segment boundaries from the label sequence
    bound_beats = rosa.util.fix_frames(bound_beats, x_min=0)  # count beat 0 as a boundary
    bound_segs = list(seg_ids[bound_beats])  # compute the segment label for each boundary
    bound_frames = beats[bound_beats]  # convert beat indices to frames
    bound_frames = rosa.util.fix_frames(bound_frames, x_min=None, x_max=C.shape[1] - 1)
    bound_times = rosa.frames_to_time(bound_frames)
    if bound_times[0] != 0:
        bound_times[0] = 0

    if plot:
        freqs = rosa.cqt_frequencies(n_bins=C.shape[0], fmin=rosa.note_to_hz("C1"), bins_per_octave=BINS_PER_OCTAVE)
        fig, ax = plt.subplots()
        colors = plt.get_cmap("Paired", k)
        rosa.display.specshow(C, y_axis="cqt_hz", sr=sr, bins_per_octave=BINS_PER_OCTAVE, x_axis="time", ax=ax)
        for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
            ax.add_patch(
                patches.Rectangle(
                    (interval[0], freqs[0]), interval[1] - interval[0], freqs[-1], facecolor=colors(label), alpha=0.50
                )
            )
        # plt.show()
        plt.savefig("workspace/laplacian_segmentation.pdf")

    return list(bound_times), list(bound_segs)
