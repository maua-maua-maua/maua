import librosa as rosa
import madmom as mm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
import sklearn.cluster

from . import cache_to_workspace
from .audio import harmonic, percussive
from .postprocess import percentile_clip


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
        sig_frames = mm.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=441)
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

    onset = percentile_clip(onset, 95)

    return onset


@cache_to_workspace("volume")
def volume(audio):
    """Creates RMS envelope from audio

    Args:
        audio (np.array): Audio signal

    Returns:
        torch.tensor, shape=(n_frames,): RMS envelope
    """
    vol = rosa.feature.rms(audio)
    vol -= vol.min()
    vol /= vol.max()
    return vol


@cache_to_workspace("chroma")
def chroma(audio, sr, type="cens", nearest_neighbor=True, preharmonic=4):
    """Creates chromagram for the harmonic component of the audio

    Args:
        audio (np.array): Audio signal
        sr (int): Sampling rate of the audio
        type (str, optional): ["cens", "cqt", "stft", "deep", "clp"]. Which strategy to use to calculate the chromagram. Defaults to "cens".
        nearest_neighbor (bool, optional): Whether to post process using nearest neighbor smoothing. Defaults to True.
        preharmonic (int, optional): For harmonic source separation, higher values create more extreme separations. Defaults to 16.

    Returns:
        torch.tensor, shape=(n_frames, 12): Chromagram
    """
    if preharmonic:
        audio = harmonic(audio, sr, preharmonic)

    if type == "cens":
        ch = rosa.feature.chroma_cens(y=audio, sr=sr)
    elif type == "cqt":
        ch = rosa.feature.chroma_cqt(y=audio, sr=sr)
    elif type == "stft":
        ch = rosa.feature.chroma_stft(y=audio, sr=sr)
    elif type == "deep":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=sr)
        ch = mm.audio.chroma.DeepChromaProcessor().process(sig).T
    elif type == "clp":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=sr)
        ch = mm.audio.chroma.CLPChromaProcessor().process(sig).T
    else:
        print("chroma type not recognized, options are: [cens, cqt, deep, clp, or stft]. defaulting to cens...")
        ch = rosa.feature.chroma_cens(y=audio, sr=sr)

    if nearest_neighbor:
        ch = np.minimum(ch, rosa.decompose.nn_filter(ch, aggregate=np.median, metric="cosine"))

    return ch


@cache_to_workspace("tonnetz")
def tonnetz(audio, sr, preharmonic=4):
    if preharmonic:
        audio = harmonic(audio, sr, preharmonic)
    return rosa.feature.tonnetz(y=audio, sr=sr)


@cache_to_workspace("pitch_track")
def pitch_track(audio, sr, preharmonic=4):
    if preharmonic:
        audio = harmonic(audio, sr, preharmonic)
    pitches, magnitudes = rosa.piptrack(y=audio, sr=sr)
    average_pitch = np.average(pitches, axis=0, weights=magnitudes + 1e-8)
    return average_pitch


@cache_to_workspace("spectral_max")
def spectral_max(audio, sr, n_mels=512):
    spectrum = rosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    spectrum = np.amax(spectrum, axis=0)
    spectrum -= spectrum.min()
    spectrum /= spectrum.max()
    return spectrum


@cache_to_workspace("pitch_dominance")
def pitch_dominance(audio, sr, preharmonic=4):
    chromagram = chroma(audio, sr, preharmonic)
    chromagram_norm = chromagram / chromagram.sum(axis=0, keepdims=1)
    chromagram_sum = np.sum(chromagram_norm, axis=1)
    pitches_sorted = np.argsort(chromagram_sum)[::-1]
    return pitches_sorted


@cache_to_workspace("pulse")
def pulse(audio, sr, prior="lognorm", prepercussive=4):
    onset_env = onsets(audio, sr, prepercussive)

    if prior == "uniform":
        prior = scipy.stats.uniform(30, 300)
    elif prior == "lognorm":
        prior = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
    else:
        prior = None

    return rosa.util.normalize(rosa.beat.plp(onset_envelope=onset_env, sr=sr, prior=prior))


@cache_to_workspace("tempo")
def tempo(audio, sr, prior="lognorm", prepercussive=4):
    onset_env = onsets(audio, sr, prepercussive)

    if prior == "uniform":
        prior = scipy.stats.uniform(30, 300)
    elif prior == "lognorm":
        prior = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
    else:
        prior = None

    tempogram = rosa.feature.tempogram(onset_envelope=onset_env, sr=sr).mean(1)
    fourier_tempogram = rosa.feature.fourier_tempogram(onset_envelope=onset_env, sr=sr).mean(1)

    tempos = -np.partition(-tempogram, 3)[:3]
    fourier_tempos = -np.partition(-fourier_tempogram, 3)[:3]
    tempo = rosa.beat.tempo(onset_envelope=onset_env, sr=sr, prior=prior)

    return (tempo, *tempos, *fourier_tempos)


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
    Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5

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
