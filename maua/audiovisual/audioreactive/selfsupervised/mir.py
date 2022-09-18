import librosa as rosa
import scipy

from .features.audio import chromagram, drop_strength, mfcc, onsets, rms, spectral_contrast, spectral_flatness, tonnetz
from .features.processing import gaussian_filter, normalize
from .features.rosa.segment import laplacian_segmentation, laplacian_segmentation_rosa

AFEATFNS = [chromagram, tonnetz, mfcc, spectral_contrast, spectral_flatness, rms, drop_strength, onsets]
UNITFEATS = ["rms", "drop_strength", "onsets", "spectral_flatness"]
ALLFEATS = ["chromagram", "tonnetz", "mfcc", "spectral_contrast"] + UNITFEATS


def salience_weighted(envelope, short_sigma=5, long_sigma=80):
    if envelope.dim() > 1:
        envelope = envelope.squeeze(1)
    short = gaussian_filter(envelope, short_sigma, mode="reflect", causal=0)
    long = gaussian_filter(envelope, long_sigma, mode="reflect", causal=0)
    weighted = (short / long) ** 2 * envelope
    if weighted.dim() < 2:
        weighted = weighted.unsqueeze(1)
    return weighted


def retrieve_music_information(audio, sr, ks=[2, 4, 6, 8, 12, 16], device="cuda"):
    features = {afn.__name__: afn(audio, sr).to(device) for afn in AFEATFNS}

    onset_env = onsets(audio, sr).squeeze().numpy()
    prior = scipy.stats.lognorm(loc=0, scale=400, s=1)
    tempo = float(rosa.beat.tempo(onset_envelope=onset_env, max_tempo=240, prior=prior, ac_size=120, hop_length=1024))
    beats = list(rosa.beat.beat_track(onset_envelope=onset_env, trim=False, hop_length=1024, bpm=tempo)[1])
    if beats[0] == 0:
        del beats[0]

    segmentations = {}
    for name, feature in features.items():
        segs = laplacian_segmentation(feature, beats, ks=ks)
        for k, s in enumerate(segs):
            segmentations[(name, ks[k])] = s.argmax(1)
    n_frames = features[AFEATFNS[0].__name__].shape[0]
    for k, rosa_seg in enumerate(laplacian_segmentation_rosa(audio.numpy(), sr, n_frames, ks=ks).to(device).unbind(1)):
        segmentations[("rosa", ks[k])] = rosa_seg.to(device)

    features = {k: normalize(salience_weighted(gaussian_filter(af, sigma=2))) for k, af in features.items()}

    return features, segmentations, tempo
