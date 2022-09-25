import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Optional

import cv2
import librosa as rosa
import librosa.display
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ..GAN.wrappers.stylegan2 import StyleGAN2
from ..ops.video import VideoWriter
from .audioreactive.selfsupervised.features.rosa.segment import BINS_PER_OCTAVE, N_OCTAVES, laplacian_segmentation_rosa
from .audioreactive.selfsupervised.mir import retrieve_music_information
from .audioreactive.selfsupervised.patch import Patch
from .audioreactive.selfsupervised.sample import load_audio

WELCOME = """
Welcome to Hans' audio-reactive video synthesizer!

This script will guide you through the steps of segmenting your audio into multiple sections and then iteratively generating visuals to match each section.

At the end the parts will be stitched together and rendered out in high resolution.

You can quit the program at any time by pressing CTRL+C or by typing 'quit' into a prompt.
"""

SEGMENTATION = """
How would you like to segment the audio?

Either:
    1) type a single integer to separate into that number of segments (e.g. response '6' will split the song into 6 separate sections, some may be repeated)

    2) type a dictionary of timestamps (in seconds) and that section's label for a manual segmentation (e.g. response '{0: 0, 30: 1, 60: 2, 90: 1, 120: 0}' will split the song into an ABCBA pattern with each section starting at 0, 30, 60, 90, and 120 seconds respectively)

Once the segmentation is satisfactory, continue with response 'next'.
"""

NOT_UNDERSTOOD = """
Response not understood, make sure that you type either a single integer, a dictionary of float keys and integer values, or the strings 'next' or 'quit'.
"""

MAIN = """
Now we will generate audio-reactive interpolations for each section. The initial proposal will be random, but you can fine-tune it with the following set of commands:
    (1) more_intense
    (2) less_intense
    (3) different_style
    (4) similar_style
    (5) different_style_motion
    (6) similar_style_motion
    (7) different_structure_motion
    (8) similar_structure_motion
    (9) revert
"""

HELP = """
'help' to show this message
'next' to continue to the next section (or final render)
'quit' to exit
(1) more_intense
(2) less_intense
(3) different_style
(4) similar_style
(5) different_style_motion
(6) similar_style_motion
(7) different_structure_motion
(8) similar_structure_motion
(9) revert
"""


def play_video(path, title):
    video = cv2.VideoCapture(path)
    cv2.namedWindow(title)
    while True:
        ret, frame = video.read()
        if not ret:
            cv2.destroyWindow(title)
            break
        cv2.imshow(title, frame)


def show_segmentation(audio, sr, segmentation):
    C = rosa.cqt(y=audio, sr=sr, hop_length=1024, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)
    C = rosa.amplitude_to_db(np.abs(C), ref=np.max)
    bounds = 1 + np.flatnonzero(segmentation[:-1] != segmentation[1:])
    bounds = rosa.util.fix_frames(bounds, x_min=0)
    bound_segs = list(segmentation[bounds])
    bound_times = bounds / (sr / 1024)
    freqs = rosa.cqt_frequencies(n_bins=C.shape[0], fmin=rosa.note_to_hz("C1"), bins_per_octave=BINS_PER_OCTAVE)
    matplotlib.use("TkAgg")
    _, ax = plt.subplots(figsize=(8, 6))
    colors = plt.get_cmap("Paired", len(np.unique(bound_segs)))
    rosa.display.specshow(
        C, y_axis="cqt_hz", sr=sr, bins_per_octave=BINS_PER_OCTAVE, x_axis="time", ax=ax, hop_length=1024
    )
    for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
        ax.add_patch(
            patches.Rectangle(
                (interval[0], freqs[0]), interval[1] - interval[0], freqs[-1], facecolor=colors(label), alpha=0.50
            )
        )
    plt.show(block=False)
    plt.pause(0.1)
    return list(bound_segs), list(bound_times)


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
        if i < self.fade_frames or i + batch_size >= fade_start:
            for batch_idx, frame_idx in enumerate(range(i, i + batch_size)):
                if frame_idx == fade_start:
                    self.avg = x[batch_idx]
                if self.fade_frames < frame_idx < fade_start or self.avg is None:
                    continue
                else:
                    smooth_idx = frame_idx - fade_start if frame_idx - fade_start >= 0 else self.fade_frames + frame_idx
                    self.avg *= 1 - self.smooth_schedule[smooth_idx]
                    self.avg += x[batch_idx] * self.smooth_schedule[smooth_idx]
                    x[batch_idx] = self.avg.clone()
        return x


class HelpPrinted(Exception):
    pass


@torch.inference_mode()
def generate_interactive(
    audio_file: str,
    stylegan2_checkpoint: str,
    latent_seeds: Optional[str] = None,
    fps: float = 24,
    audio_offset: float = 0,
    audio_duration: Optional[float] = None,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    print(WELCOME)

    lo_res = (512, 512)
    preview_time = 10
    hi_res = (2048, 1024)
    fade_time = 2

    print("Loading audio...")
    audio, sr = load_audio(audio_file, audio_offset, audio_duration, fps)
    audio_duration = len(audio) / sr
    n_frames = round(audio_duration * sr / 1024)

    print(SEGMENTATION)
    response = input("> ")
    while response != "next":

        # parse response
        try:
            if response == "next" or response == "n":
                break
            elif response == "quit" or response == "q":
                exit(0)
            else:
                response = eval(response)
                assert isinstance(response, (int, dict))
        except AssertionError:
            print(NOT_UNDERSTOOD)
            response = input("> ")
            continue

        # segment automatically
        if isinstance(response, int):
            segmentation = laplacian_segmentation_rosa(audio.numpy(), sr, n_frames, ks=[response]).squeeze()

        # segment manually
        else:
            times = list(response.keys())
            labels = list(response.values())
            segmentation = []
            for start, end, label in zip(times, times[1:] + [audio_duration], labels):
                segmentation.append(torch.ones(round(end * fps - start * fps)) * label)
            segmentation = torch.cat(segmentation)

        # solicit feedback
        labels, times = show_segmentation(audio.numpy(), sr, segmentation.numpy())
        response = input("> ")
        plt.close()

    label_vals = np.unique(labels)
    first_times = [labels.index(s) for s in label_vals]
    times.append(audio_duration)
    segments = [(lbl, times[ft], times[ft + 1]) for lbl, ft in zip(label_vals, first_times)]
    segments = sorted(segments, key=lambda s: s[0])

    print(MAIN)

    print("Extracting musical features from audio...")
    features, segmentations, tempo = retrieve_music_information(audio, sr)
    G = StyleGAN2(model_file=stylegan2_checkpoint, output_size=lo_res).to(device)
    final = {}

    for s, (label, start, end) in enumerate(segments):

        print(f"Segment {s+1}: {start} - {end}")

        sf, ef = round(start * fps), round(end * fps)
        feats = {k: feat[sf:ef] for k, feat in features.items()}
        segs = {k: seg[sf:ef] for k, seg in segmentations.items()}

        seed = torch.randint(0, 2**32, size=(), device=device).item()
        r, response, patch, prev_patches, prev_palettes = 1, "start", None, [], []
        intensity = 0.666

        while response != "next":

            try:
                # parse response
                if response == "next" or response == "n":
                    break
                elif response == "quit" or response == "q":
                    exit(0)

                # update patch
                if patch is None:
                    patch = Patch(features=feats, segmentations=segs, tempo=tempo, seed=seed, fps=fps, device=device)

                    if latent_seeds is None:
                        z = torch.randn((20, 512), device=device, generator=torch.Generator(device).manual_seed(seed))
                        latent_palette = G.mapper(z)
                    else:
                        latent_palette = G.get_w_latents(latent_seeds)

                    print("Initial random patch:")
                else:
                    for command in response.split(","):
                        if command in ["1", "more_intense"]:
                            intensity += 0.111
                            patch.update_intensity(intensity)
                        if command in ["2", "less_intense"]:
                            intensity -= 0.111
                            patch.update_intensity(intensity)

                        if command in ["3", "different_style"]:
                            latent_palette = G.mapper(torch.randn((20, 512), device=device))
                        if command in ["4", "similar_style"]:
                            latent_palette = latent_palette[np.random.permutation(latent_palette.shape[0])]
                        if command in ["5", "different_style_motion"]:
                            patch.randomize_latent_patches()
                        if command in ["6", "similar_style_motion"]:
                            patch.latent_patches = list(np.random.permutation(patch.latent_patches))
                        if command in ["7", "different_structure_motion"]:
                            patch.randomize_noise_patches()
                        if command in ["8", "similar_structure_motion"]:
                            patch.noise_patches = list(np.random.permutation(patch.noise_patches))

                        if command in ["9", "revert"]:
                            patch = prev_patches.pop()
                            latent_palette = prev_palettes.pop()

                        if command == "help":
                            print(HELP)
                            raise HelpPrinted()

                    print("Updated patch:")
                prev_patches.append(deepcopy(patch))
                prev_palettes.append(latent_palette.clone())
                print(patch)

                latents, noise = patch.forward(latent_palette, downscale_factor=2)

                # render video
                out_file = f"output/{Path(audio_file).stem}_RandomPatches++_seed{seed}_{r}.mp4"
                r += 1
                with VideoWriter(
                    output_file=out_file,
                    output_size=lo_res,
                    fps=fps,
                    audio_file=audio_file,
                    audio_offset=start,
                    audio_duration=min(preview_time, end - start),
                ) as video:
                    for i in tqdm(
                        range(0, len(latents[: round(preview_time * fps)]) - batch_size, batch_size),
                        unit_scale=batch_size,
                    ):
                        L = latents[i : i + batch_size]

                        N = {}
                        for j, noise_module in enumerate(noise):
                            N[f"noise{j}"] = noise_module.forward(i, batch_size)[:, None]

                        for frame in G.synthesizer(latents=L, **N).add(1).div(2):
                            video.write(frame.unsqueeze(0))

                        if i == 0:
                            patch_file = out_file.replace(".mp4", ".json")
                            patch.save(patch_file)

                play_video(out_file, title=f"Segment {s+1}: {start} - {end}, version {r}")
            except HelpPrinted:
                continue
            except:
                print("\n\nERROR!")
                traceback.print_exc()
                print("\n")
            response = input("> ")

        final[label] = deepcopy(patch), latent_palette.clone()

    print("Rendering final video...")
    del G
    G = StyleGAN2(model_file=stylegan2_checkpoint, output_size=hi_res, strategy="pad-reflect-out").to(device)
    out_file = f"output/{Path(stylegan2_checkpoint).stem}_{Path(audio_file).stem}_final.mp4"
    batch_size = round(batch_size / 4)
    fade_frames = fade_time * fps
    latent_fade, noise_fades = EMAFade(fade_frames), [EMAFade(fade_frames) for _ in noise]
    with VideoWriter(output_file=out_file, output_size=hi_res, fps=fps, audio_file=audio_file) as video:
        for label, start, end in zip(tqdm(labels, desc="Sections"), times[:-1], times[1:]):
            patch, palette = final[label]
            latents, noise = patch.forward(palette)
            for i in tqdm(range(0, len(latents) - batch_size, batch_size), unit_scale=batch_size, desc="Frames"):
                L = latent_fade(latents[i : i + batch_size], i, len(latents))

                N = {}
                for j, (noise_fade, noise_module) in enumerate(zip(noise_fades, noise)):
                    N[f"noise{j}"] = noise_fade(noise_module.forward(i, batch_size)[:, None], i, len(latents))

                for frame in G.synthesizer(latents=L, **N).add(1).div(2):
                    video.write(frame.unsqueeze(0))


if __name__ == "__main__":
    import fire

    fire.Fire(generate_interactive)
