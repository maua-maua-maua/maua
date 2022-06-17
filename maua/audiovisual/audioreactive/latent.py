import torch
from torch.nn.functional import interpolate
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from torchtyping import TensorType

from scipy import signal

# from . import cache_to_workspace
from .signal import gaussian_filter, normalize, resample


# @cache_to_workspace("single_weighted")
def single_weighted(
    low_latent: TensorType["n_layers", "latent_dim"],
    high_latent: TensorType["n_layers", "latent_dim"],
    envelope: TensorType["time"],
) -> TensorType["time", "n_layers", "latent_dim"]:
    return low_latent[None] * (1 - envelope[:, None, None]) + high_latent[None] * envelope[:, None, None]


# @cache_to_workspace("multi_weighted")
def multi_weighted(
    latents: TensorType["n_latents", "n_layers", "latent_dim"], envelopes: TensorType["time", "n_latents"]
) -> TensorType["time", "n_layers", "latent_dim"]:
    envelopes /= envelopes.sum(dim=1, keepdim=True)
    latents = torch.einsum(
        "Atwl,Atwl->twl",
        envelopes.permute(1, 0)[..., None, None],
        latents[torch.arange(envelopes.shape[1]) % len(latents), None],
    )
    return latents


# @cache_to_workspace("select_modulo")
def select_modulo(
    latents: TensorType["n_latents", "n_layers", "latent_dim"], envelope: TensorType["time"], smooth: float = 2
) -> TensorType["time", "n_layers", "latent_dim"]:
    low, high = torch.quantile(envelope, 0.25), torch.quantile(envelope, 0.75)
    indices = normalize(envelope.clamp(low, high))
    indices *= len(latents) - 1
    indices = indices.round().long()
    out = torch.from_numpy(latents.cpu().numpy()[indices.cpu()]).to(latents)  # TODO how can I do this with torch?
    out = gaussian_filter(out, smooth, causal=0)
    return out


def eerp(a, b, t):
    return a ** (1 - t) * b**t


def copeerp(a, b, t):
    return a**t * (1 - b**t) / (1 - a**t + b**t)


def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    p = p.permute(2, 0, 1)[..., None]
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a[None] * torch.cos(p) + c[None] * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d


# @cache_to_workspace("slerp_loops")
def slerp_loops(
    y: TensorType["n_latents", "n_layers", "latent_dim"], size: TensorType["size", int], n_loops: int
) -> TensorType["size", "n_layers", "latent_dim"]:
    y = torch.cat([y] * n_loops + [y[[0]]])
    t = torch.linspace(0, 1, round(size / len(y))).to(y)
    y = y.unbind(0)
    ya, yb = torch.stack(y[:-1]), torch.stack(y[1:])
    out = slerp(ya, yb, t)
    out = out.reshape(-1, *out.shape[2:])
    out = interpolate(out.permute(1, 2, 0), size=size, mode="linear", align_corners=False)
    out = out.permute(2, 0, 1)
    return out


# @cache_to_workspace("spline_loops")
def spline_loops(
    y: TensorType["n_latents", "n_layers", "latent_dim"], size: TensorType["size", int], n_loops: int
) -> TensorType["size", "n_layers", "latent_dim"]:
    y = torch.cat([y] * n_loops + [y[[0]]])
    t_in = torch.linspace(0, 1, len(y)).to(y)
    t_out = torch.linspace(0, 1, size).to(y)
    coeffs = natural_cubic_spline_coeffs(t_in, y.permute(1, 0, 2))
    out = NaturalCubicSpline(coeffs).evaluate(t_out)
    return out.permute(1, 0, 2)


# @cache_to_workspace("tempo_loops")
def tempo_loops(latents, n_frames, fps, tempo, type="spline"):
    bars_per_sec = tempo / 4 / 60
    duration = n_frames / fps
    n_loops = round(duration * bars_per_sec)
    if type == "spline":
        return spline_loops(latents, n_frames, n_loops)
    else:
        return slerp_loops(latents, n_frames, n_loops)


if __name__ == "__main__":
    with torch.inference_mode():
        from ...ops.video import VideoWriter
        from ...GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer
        from .audio import load_audio
        from .mir import onsets, chroma, tempo
        from tqdm import tqdm

        duration = 120
        fps = 24
        n_frames = fps * duration
        batch_size = 8
        size = 512
        n_latents = 12

        audio_file = "/home/hans/datasets/wavefunk/naamloos.wav"
        audio, sr, duration = load_audio(audio_file, duration=duration)
        audio = audio.numpy()

        ckpt = "/home/hans/modelzoo/wavefunk/cyphept-1024-045000.pt"
        mapper = StyleGAN2Mapper(ckpt, False)
        synthesizer = StyleGAN2Synthesizer(ckpt, False, (size, size), "stretch", 0)
        mapper, synthesizer = mapper.cuda(), synthesizer.cuda()

        latents = mapper(torch.randn((n_latents, 512), device="cuda"))

        onset_env = torch.from_numpy(signal.resample(onsets(audio, sr), n_frames)).cuda()
        chroma_envs = torch.from_numpy(signal.resample(torch.from_numpy(chroma(audio, sr)), n_frames)).cuda()
        main_tempo = tempo(audio, sr)[0]

        single_weighted_latents = single_weighted(latents[0], latents[1], onset_env)
        multi_weighted_latents = multi_weighted(latents, chroma_envs)
        select_modulo_latents = select_modulo(latents, onset_env)
        slerp_loops_latents = slerp_loops(latents, n_frames, n_loops=3)
        spline_loops_latents = spline_loops(latents, n_frames, n_loops=3)
        tempo_loops_latents = tempo_loops(latents, n_frames, fps, main_tempo)

        for name, lats in [
            ("single_weighted_latents", single_weighted_latents),
            ("multi_weighted_latents", multi_weighted_latents),
            ("select_modulo_latents", select_modulo_latents),
            ("slerp_loops_latents", slerp_loops_latents),
            ("spline_loops_latents", spline_loops_latents),
            ("tempo_loops_latents", tempo_loops_latents),
        ]:
            with VideoWriter(
                output_file=f"output/test_{name}.mp4",
                output_size=(size, size),
                fps=fps,
                audio_file=audio_file,
                audio_duration=duration,
            ) as video:
                for batch in tqdm(
                    gaussian_filter(lats, 1, causal=0).split(batch_size), desc=name, unit_scale=batch_size
                ):
                    for frame in synthesizer(batch).add(1).div(2):
                        video.write(frame[None])
