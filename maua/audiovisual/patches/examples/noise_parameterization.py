import random

import numpy as np
import scipy.ndimage as ndi
import torch
from kornia.geometry.transform import rotate
from scipy import signal

from ... import audioreactive as ar
from ...patches.base.stylegan2 import StyleGAN2Patch


def random_selection(latents, n):
    return latents[torch.randperm(latents.shape[0])[:n]]


def circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius

    return torch.from_numpy(mask).float()


def postprocess(x, n_frames, smooth):
    return ar.normalize(
        torch.from_numpy(
            signal.resample(
                ar.gaussian_filter(
                    x.clamp(torch.quantile(x, 0.1), torch.quantile(x, 0.975)), smooth, causal=0.1
                ).numpy(),
                n_frames,
            )
        )
    )


class NoiseParameterization(StyleGAN2Patch):
    def process_audio(self):
        print(self.n_frames)
        self.onsets = ar.onsets(self.audio, self.sr).reshape(-1, 1, 1)
        self.onsets = postprocess(self.onsets, self.n_frames, smooth=40)

        self.volume = ar.volume(self.audio, self.sr).reshape(-1, 1, 1)
        self.volume = postprocess(self.volume, self.n_frames, smooth=80)

        self.chroma = ar.tonnetz(self.audio, self.sr)
        self.chroma = postprocess(self.chroma, self.n_frames, smooth=20)

    def process_mapper_inputs(self):
        return {"latent_z": self.stylegan2.get_z_latents("1-40,400-440")}

    def process_synthesizer_inputs(self, latent_w):
        base_structure = random_selection(latent_w[:40], 10)
        chroma_colors = random_selection(latent_w[40:], 6)
        onset_colors = random_selection(latent_w[40:], random.choice(range(3, 7)))
        volume_colors = random_selection(latent_w[40:], random.choice(range(3, 7)))

        chroma_latents = ar.chroma_weighted(chroma_colors, self.chroma)
        base_loop = ar.spline_loops(base_structure, self.n_frames, n_loops=random.choice(range(1, 3)))
        onset_latents = ar.spline_loops(onset_colors, self.n_frames, n_loops=random.choice(range(2, 7)))
        volume_latents = ar.spline_loops(volume_colors, self.n_frames, n_loops=random.choice(range(2, 7)))

        latents = chroma_latents
        latents[:, :4] = base_loop[:, :4]
        latents = (1 - self.volume) * latents + self.volume * volume_latents
        latents = (1 - self.onsets) * latents + self.onsets * onset_latents

        latents = ar.gaussian_filter(latents, 2)

        n_rotations = self.n_frames / random.choice([6 * fps, 6.5 * fps, 7 * fps, 8 * fps])
        steps_per_rev = int(self.n_frames / n_rotations)
        revolution = -torch.linspace(0, 360 * (1 - 1 / steps_per_rev), steps_per_rev)
        angles = torch.cat([revolution for _ in range(int(self.n_frames / steps_per_rev))])
        if self.n_frames - len(angles) > 0:
            angles = torch.cat([angles, revolution[: self.n_frames - len(angles)]])  # pad any rounding errors
        angles = angles[: self.n_frames]

        s = 64
        time = random.choice([4, 8])
        space = random.choice([4, 8])
        rotating_noise = rotate(
            ar.perlin_noise((self.n_frames, s, s), (time, space, space)).unsqueeze(1), angles, padding_mode="reflection"
        )
        static_noise = ar.perlin_noise((self.n_frames, s, s), (time, space, space)).unsqueeze(1)

        disc_mask = circular_mask(s, s) - circular_mask(s, s, radius=int(s / random.choice([6, 6.5, 7])))

        noise = (1 - disc_mask) * static_noise + random.choice([1, 2, 3, 4]) * disc_mask * rotating_noise
        noise = torch.from_numpy(ndi.gaussian_filter(noise.numpy(), [0, 0, 1, 1]))
        noise -= noise.mean(dim=(2, 3), keepdim=True)
        noise /= ar.gaussian_filter(noise.std(dim=(2, 3), keepdim=True), 10)
        noise *= random.choice([1, 2, 3, 4])

        max_layer = 13
        noises = self.synthesizer.make_noise_pyramid(noise, layer_limit=max_layer)
        noises["noise0"] = ar.gaussian_filter(torch.randn_like(noises["noise0"]), 50)
        noises["noise0"] /= ar.gaussian_filter(noises["noise0"].std(dim=(2, 3), keepdim=True), 10)
        noises["noise1"] += random.choice([2, 4]) * ar.gaussian_filter(torch.randn_like(noises["noise1"]), 50)
        noises["noise1"] /= ar.gaussian_filter(noises["noise1"].std(dim=(2, 3), keepdim=True), 10)
        noises["noise2"] += random.choice([1, 2]) * ar.gaussian_filter(torch.randn_like(noises["noise2"]), 50)
        noises["noise2"] /= ar.gaussian_filter(noises["noise2"].std(dim=(2, 3), keepdim=True), 10)

        class LoopingTensor(torch.Tensor):
            def __getitem__(self, indices):
                if isinstance(indices, slice):
                    indices = torch.arange(indices.start, indices.stop, indices.step) % self.shape[0]
                elif isinstance(indices, int):
                    indices = indices % self.shape[0]
                return super().__getitem__(indices)

            def size(self, idx=None):
                size = torch.Size((len(latents), *self.shape[1:]))
                return size[idx] if idx is not None else size

        for l in range(max_layer, 17):
            size = 2 ** (2 + (l + 1) // 2)
            noises[f"noise{l}"] = ar.gaussian_filter(torch.randn((128, 1, size, size)), random.choice([1, 2]))
            noises[f"noise{l}"] /= ar.gaussian_filter(noises[f"noise{l}"].std(dim=(2, 3), keepdim=True), 20)
            noises[f"noise{l}"] *= random.choice([1, 1, 1.5])
            noises[f"noise{l}"] = LoopingTensor(noises[f"noise{l}"])

        return {"latents": latents, **noises}
