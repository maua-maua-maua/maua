import librosa as rosa
import scipy

from .features.audio import chromagram, drop_strength, mfcc, onsets, rms, spectral_contrast, spectral_flatness, tonnetz
from .features.processing import gaussian_filter, normalize
from .features.rosa.segment import laplacian_segmentation, laplacian_segmentation_rosa

UNIT_FEATURES = [rms, drop_strength, onsets, spectral_flatness]
AUDIO_FEATURES = [chromagram, tonnetz, mfcc, spectral_contrast] + UNIT_FEATURES


def salience_weighted(envelope, short_sigma=5, long_sigma=80):
    if envelope.dim() > 1:
        envelope = envelope.squeeze(1)
    short = gaussian_filter(envelope, short_sigma, mode="reflect", causal=0)
    long = gaussian_filter(envelope, long_sigma, mode="reflect", causal=0)
    weighted = (short / long) ** 2 * envelope
    if weighted.dim() < 2:
        weighted = weighted.unsqueeze(1)
    return weighted


def estimate_tempo(audio, sr):
    onset_env = onsets(audio, sr).squeeze().cpu().numpy()
    tempo = rosa.beat.tempo(
        onset_envelope=onset_env,
        max_tempo=240,
        prior=scipy.stats.lognorm(loc=0, scale=400, s=1),
        ac_size=120,
        hop_length=1024,
    )
    return float(tempo)


def estimate_beats(audio, sr, tempo=None):
    tempo = estimate_tempo(audio, sr) if tempo is None else tempo
    onset_env = onsets(audio, sr).squeeze().cpu().numpy()
    beats = list(rosa.beat.beat_track(onset_envelope=onset_env, trim=False, hop_length=1024, bpm=tempo)[1])
    if beats[0] == 0:
        del beats[0]
    return beats


def retrieve_music_information(audio, sr, ks=[2, 4, 6, 8, 12, 16], device="cuda"):
    features = {afn.__name__: afn(audio, sr).to(device) for afn in AUDIO_FEATURES}
    tempo = estimate_tempo(audio, sr)
    beats = estimate_beats(audio, sr)

    segmentations = {}
    for name, feature in features.items():
        segs = laplacian_segmentation(feature, beats, ks=ks)
        for k, s in enumerate(segs):
            segmentations[(name, ks[k])] = s.argmax(1)
    n_frames = features[AUDIO_FEATURES[0].__name__].shape[0]
    for k, rosa_seg in enumerate(
        laplacian_segmentation_rosa(audio.cpu().numpy(), sr, n_frames, ks=ks).to(device).unbind(1)
    ):
        segmentations[("rosa", ks[k])] = rosa_seg.to(device)

    features = {k: normalize(salience_weighted(gaussian_filter(af, sigma=2))) for k, af in features.items()}

    return features, segmentations, tempo
