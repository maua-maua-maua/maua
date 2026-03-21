from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from maua.audiovisual.audioreactive.selfsupervised.sample import load_audio
from maua.GAN.wrappers.stylegan2 import StyleGAN2
from maua.ops.video import VideoWriter

torch.backends.cudnn.allow_tf32 = True
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class EMAFade(torch.nn.Module):
    # TODO this fading strategy is very brittle, especially for short sections

    def __init__(self, fade_frames) -> None:
        super().__init__()
        self.fade_frames = fade_frames
        self.smooth_schedule = torch.cat((torch.linspace(1, 0, fade_frames), torch.linspace(0, 1, fade_frames)))
        self.avg = None

    def forward(self, x, i, total_length):
        batch_size = x.shape[0]
        fade_start = total_length - self.fade_frames
        fade_end = self.fade_frames
        if i < fade_end or i + batch_size >= fade_start:
            for batch_idx, frame_idx in enumerate(range(i, i + batch_size)):
                if frame_idx == fade_start:
                    self.avg = x[batch_idx].clone()
                if fade_end < frame_idx < fade_start or self.avg is None:
                    continue
                else:
                    smooth_idx = frame_idx - fade_start if frame_idx - fade_start >= 0 else fade_end + frame_idx
                    if smooth_idx >= len(self.smooth_schedule):
                        continue
                    self.avg *= 1 - self.smooth_schedule[smooth_idx]
                    self.avg += x[batch_idx] * self.smooth_schedule[smooth_idx]
                    x[batch_idx] = self.avg.clone()
        return x


class Section(torch.nn.Module):
    def __init__(self, start, latent, noise, seeds=None, tempo=None):
        super().__init__()
        self.start = start
        self.seeds = seeds
        self.latent = latent
        self.noise = noise
        self.tempo = tempo


@torch.inference_mode()
def audio_reactive_interpolation(
    sections,
    audio_file: str,
    auxiliary_audio: dict[str, str] | None = None,
    stylegan2_checkpoint: str = "/home/hans/modelzoo/koanGAN/select/koancept.pkl",
    render_sections: slice | None = None,
    aspect_ratio: float = 2,
    height: int = 1024,
    fade_time: float = 2,
    fps: float = 24,
    truncation: float = 1.0,
    batch_size: int = 16,
    resize_strategy: str = "pad-reflect-out",
    resize_layer: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    out_dir="output/",
):
    # preprocess inputs
    out_file = f"{out_dir}/{Path(audio_file).stem} | {Path(stylegan2_checkpoint).stem} | {datetime.now()}"

    audio, sr = load_audio(audio_file, 0, None, fps)
    if auxiliary_audio is not None:
        auxiliary_audio = {name: load_audio(aux_file, 0, None, fps)[0] for name, aux_file in auxiliary_audio.items()}
    duration = len(audio) / sr

    fade_frames = round(fade_time * fps)

    times = [section.start for section in sections] + [duration]
    times = list(zip(times[:-1], times[1:]))
    if render_sections is not None:
        sections = sections[render_sections]
        times = times[render_sections]
    sections = torch.nn.ModuleList(sections).to(device)

    width_height = (round(aspect_ratio * height), height)
    G = StyleGAN2(
        model_file=stylegan2_checkpoint, output_size=width_height, strategy=resize_strategy, layer=resize_layer
    ).to(device)

    # prepare audio features for each section
    for section, (start, end) in zip(sections, times):
        ss, es = round(start * sr), round(end * sr)
        section_audio = audio[ss:es].to(device)
        section_aux = None
        if auxiliary_audio is not None:
            section_aux = {name: aux[ss:es].to(device) for name, aux in auxiliary_audio.items()}

        for latent_patch in section.latent.modules():
            if hasattr(latent_patch, "seeds") and latent_patch.seeds is None and section.seeds is not None:
                latent_patch.seeds = section.seeds
            if hasattr(latent_patch, "tempo") and latent_patch.tempo is None and section.tempo is not None:
                latent_patch.tempo = section.tempo
            latent_patch.prepare(section_audio, sr, section_aux)
            latent_patch.to(device)
            if hasattr(latent_patch, "palette"):
                latent_patch.palette = G.mapper(latent_patch.palette, truncation=truncation)

        for noise_patch in section.noise.modules():
            if hasattr(noise_patch, "tempo") and noise_patch.tempo is None and section.tempo is not None:
                noise_patch.tempo = section.tempo
            noise_patch.prepare(section_audio, sr, section_aux)
            noise_patch.to(device)

    # render full video section by section
    with (
        VideoWriter(
            output_file=f"{out_file}.mp4",
            output_size=G.synthesizer.output_size,
            fps=fps,
            audio_file=audio_file,
            audio_offset=times[0][0],
            audio_duration=times[-1][1] - times[0][0],
        ) as video,
        tqdm(total=round((times[-1][1] - times[0][0] + 1) * fps), unit="frames") as progress,
    ):
        latent_fade, noise_fades = EMAFade(fade_frames), [EMAFade(fade_frames) for _ in range(20)]
        for s, section in enumerate(sections):
            latents = section.latent.forward()
            progress.set_description(f"Rendering section {s}...")
            for i in range(0, len(latents), batch_size):
                L = latent_fade(latents[i : i + batch_size], i, len(latents))
                N = {
                    f"noise{j}": noise_fade(noise.unsqueeze(1), i, len(latents))
                    for j, (noise_fade, noise) in enumerate(
                        zip(noise_fades, section.noise.forward(i, batch_size, 1024 / height, aspect_ratio))
                    )
                }
                for frame in G.synthesizer(latents=L, **N).add(1).div(2):
                    video.write(frame.unsqueeze(0))
                progress.update(batch_size)
