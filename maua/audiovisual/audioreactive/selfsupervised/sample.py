import sys
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm

from ....GAN.wrappers.stylegan2 import StyleGAN2
from ....ops.video import VideoWriter
from .mir import retrieve_music_information
from .patch import Patch


def load_audio(audio_file, offset, duration, fps):
    audio, sr = torchaudio.load(audio_file)

    # convert to mono
    audio = audio.mean(0)

    # extract specified portion of audio
    if duration is not None:
        audio = audio[offset * sr : (offset + duration) * sr]
    else:
        audio = audio[offset * sr :]

    # resample to correct sampling rate for specified fps
    new_sr = 1024 * fps
    audio = resample(audio, sr, new_sr)

    return audio, new_sr


@torch.inference_mode()
def generate(
    audio_file: str,
    stylegan2_checkpoint: str,
    patch_file: Optional[str] = None,
    seed: int = None,
    latent_seeds: Optional[str] = None,
    fps: float = 30,
    audio_offset: float = 0,
    audio_duration: Optional[float] = None,
    downscale_factor: float = 4,
    aspect_ratio: float = 1,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if seed is None:
        seed = torch.randint(0, 2**32, size=(), device=device).item()

    out_size = (round(aspect_ratio * 1024 / downscale_factor), round(1024 / downscale_factor))
    out_file = f"output/{Path(audio_file).stem}_RandomPatches++_seed{seed}_{out_size[0]}x{out_size[1]}.mp4"

    print("extracting information from audio...")
    audio, sr = load_audio(audio_file, audio_offset, audio_duration, fps)
    features, segmentations, tempo = retrieve_music_information(audio, sr)

    if patch_file is None:
        print("generating random audioreactive patch...")
        patch = Patch(features=features, segmentations=segmentations, tempo=tempo, seed=seed, fps=fps, device=device)
    else:
        print("loading audioreactive patch from file...")
        patch = Patch.load(
            patch_file, features=features, segmentations=segmentations, tempo=tempo, fps=fps, device=device
        )
    print(patch)

    G = StyleGAN2(model_file=stylegan2_checkpoint, output_size=out_size).to(device)

    if latent_seeds is None:
        z = torch.randn((180, 512), device=device, generator=torch.Generator(device).manual_seed(seed))
        latent_palette = G.mapper(z)
    else:
        latent_palette = G.get_w_latents(latent_seeds)

    print("preparing latent and noise sequences...")
    latents, noise = patch.forward(latent_palette, downscale_factor=downscale_factor, aspect_ratio=aspect_ratio)

    print("rendering...")
    with VideoWriter(
        output_file=out_file,
        output_size=out_size,
        fps=fps,
        audio_file=audio_file,
        audio_offset=audio_offset,
        audio_duration=audio_duration,
    ) as video:
        for i in tqdm(range(0, len(latents) - batch_size, batch_size), unit_scale=batch_size):
            L = latents[i : i + batch_size]

            N = {}
            for j, noise_module in enumerate(noise):
                N[f"noise{j}"] = noise_module.forward(i, batch_size)[:, None]

            for frame in G.synthesizer(latents=L, **N).add(1).div(2):
                video.write(frame.unsqueeze(0))

            if i == 0:
                patch.save(out_file.replace(".mp4", ".json"))


if __name__ == "__main__":
    import fire

    fire.Fire(generate)
