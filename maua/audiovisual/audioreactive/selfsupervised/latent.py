from enum import Enum
from typing import Union

import numpy as np
import torch
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from .features.audio import onsets
from .features.processing import gaussian_filter, normalize
from .mir import chromagram, estimate_beats, estimate_tempo, laplacian_segmentation


def spline_loop_latents(y, size, n_loops=1):
    y = torch.cat((y, y[[0]]))
    t_in = torch.linspace(0, 1, len(y)).to(y)
    t_out = torch.linspace(0, n_loops, size).to(y) % 1
    coeffs = natural_cubic_spline_coeffs(t_in, y.permute(1, 0, 2))
    out = NaturalCubicSpline(coeffs).evaluate(t_out)
    return out.permute(1, 0, 2)


class Latents(torch.nn.Module):
    def __init__(self, seeds) -> None:
        super().__init__()
        self.seeds = seeds

    def prepare(self):
        if self.seeds is None:
            self.seeds = torch.randint(2**32 - 1, size=(32,))
        palette = np.concatenate([np.random.RandomState(int(seed)).randn(1, 512) for seed in self.seeds])
        self.register_buffer("palette", torch.FloatTensor(palette))


class SegmentationLatents(Latents):
    def __init__(self, feature=chromagram, segments=4, sigma=5, seeds=None, tempo=None):
        super().__init__(seeds)
        self.feature = feature
        self.segments = segments
        self.sigma = sigma
        self.tempo = tempo

    def prepare(self, audio, sr, auxiliary=None):
        super().prepare()
        modulation = laplacian_segmentation(
            normalize(gaussian_filter(self.feature(audio, sr), 2)),
            estimate_beats(audio, sr, self.tempo),
            [self.segments],
        )[0].argmax(1)
        self.register_buffer("modulation", modulation)

    def forward(self):
        return gaussian_filter(self.palette[self.modulation.cpu().numpy()], self.sigma)


class FeatureLatents(Latents):
    def __init__(self, feature, smooth=2, seeds=None, focus=None):
        super().__init__(seeds)
        self.feature = feature
        self.focus = focus
        self.smooth = smooth

    def prepare(self, audio, sr, auxiliary=None):
        if self.focus is not None:
            audio = auxiliary[self.focus]

        modulation = normalize(gaussian_filter(self.feature(audio, sr), self.smooth))

        if len(modulation.shape) == 1 or modulation.shape[1] == 1:
            modulation = torch.stack((modulation, 1 - modulation), dim=1).squeeze()

        self.register_buffer("modulation", modulation)
        super().prepare()

    def forward(self):
        return torch.einsum(
            "TN,NWL->TWL",
            self.modulation / self.modulation.sum(1, keepdim=True),
            self.palette[: self.modulation.shape[1]],
        )


class CumulativeFeatureLatents(Latents):
    def __init__(self, feature, seeds=None, smooth=1, focus=None, friction=5):
        super().__init__(seeds)
        self.feature = feature
        self.smooth = smooth
        self.focus = focus
        self.slow_down = friction

    def prepare(self, audio, sr, auxiliary=None):
        super().prepare()

        if self.focus is not None:
            audio = auxiliary[self.focus]

        modulation = normalize(gaussian_filter(self.feature(audio, sr), self.smooth, causal=0))
        if len(modulation.shape) != 1:
            modulation = modulation[:, 0]

        cumulative = (torch.cumsum(modulation, dim=0) / self.slow_down) % (len(self.palette) + 1)

        self.register_buffer("cumulative", cumulative)

    def forward(self):
        y = torch.cat((self.palette, self.palette[[0]]))
        t_in = torch.linspace(0, len(y), len(y)).to(y)
        t_out = self.cumulative.to(y)
        coeffs = natural_cubic_spline_coeffs(t_in, y.permute(1, 0, 2))
        out = NaturalCubicSpline(coeffs).evaluate(t_out).permute(1, 0, 2)
        return out


class LoopLatents(Latents):
    def __init__(self, n_latents=None, seeds=None, n_loops=None, loop_bars=None, speed=None, tempo=None):
        super().__init__(seeds)
        self.n_latents = n_latents

        assert n_loops is not None or loop_bars is not None or speed is not None
        self.speed = speed
        self.n_loops = n_loops
        self.loop_bars = loop_bars

        self.tempo = tempo

    def prepare(self, audio, sr, auxiliary=None):
        super().prepare()
        if self.speed is not None:
            self.n_latents = max(2, round(self.speed * len(audio) / sr))
            self.n_loops = 1
        if self.n_latents is not None:
            self.palette = self.palette[: self.n_latents]
        if self.tempo is None:
            self.tempo = estimate_tempo(audio, sr)
        self.n_frames = len(onsets(audio, sr))
        self.n_loops = self.n_loops or len(audio) / sr / 60 * self.tempo / 4 / self.loop_bars

    def forward(self):
        return spline_loop_latents(y=self.palette, size=self.n_frames, n_loops=self.n_loops)


class SmoothRandomLatents(Latents):
    def __init__(self, sigma, seed=42):
        super().__init__(None)
        self.sigma = sigma
        self.seed = seed

    def prepare(self, audio, sr, auxiliary=None):
        self.register_buffer(
            "palette",
            gaussian_filter(
                torch.randn((len(onsets(audio, sr)), 512), generator=torch.Generator().manual_seed(self.seed)),
                self.sigma,
            ),
        )

    def forward(self):
        return self.palette


class ConstantLatents(Latents):
    def __init__(self, seeds=None, latent_file=None) -> None:
        super().__init__(seeds)
        self.latent_file = latent_file

    def prepare(self, audio, sr, auxiliary=None):
        super().prepare()
        self.n_frames = len(onsets(audio, sr))
        if self.latent_file is not None:
            self.mapped_latent = torch.load(self.latent_file)["latent"]

    def forward(self):
        if self.latent_file is None:
            return self.palette[0].tile(self.n_frames, 1, 1)
        else:
            return self.mapped_latent.tile(self.n_frames, 1, 1)


class VelocityLatents(Latents):
    def __init__(self, feature, multiplier=1, seeds=None, smooth=6, focus=None) -> None:
        super().__init__(seeds)
        self.multiplier = multiplier
        self.smooth = smooth
        self.feature = feature
        self.focus = focus

    def prepare(self, audio, sr, auxiliary=None):
        super().prepare()

        if self.focus is not None:
            audio = auxiliary[self.focus]

        self.velocity = normalize(gaussian_filter(self.feature(audio, sr), self.smooth, causal=1)).cpu()

        t_delta = 1 / len(self.velocity)

        latents = []
        current = self.palette[0]
        direction = spline_loop_latents(y=self.palette[1:, None], size=len(self.velocity), n_loops=1).squeeze()
        for i in range(len(self.velocity)):
            current += self.multiplier * self.velocity[i] * t_delta * direction[i]
            latents.append(current.clone())

        self.palette = torch.stack(latents)

    def forward(self):
        return self.palette


class PitchLatents(Latents):
    pass


class LucidLatents(Latents):
    pass


class MergeDepth(Enum):
    LOW: slice = slice(0, 6)
    MID: slice = slice(6, 12)
    HIGH: slice = slice(12, 18)
    LOWMID: slice = slice(0, 12)
    MIDHIGH: slice = slice(6, 18)
    ALL: slice = slice(0, 18)


class MergeLatents(Latents):
    def __init__(self, left: Latents, right: Latents, depth: Union[MergeDepth, slice] = MergeDepth.ALL):
        super().__init__(seeds=None)
        self.left = left
        self.right = right
        self.slice = depth.value

    def prepare(self, audio, sr, auxiliary=None):
        pass


class OverwriteLatents(MergeLatents):
    def forward(self):
        left, right = self.left(), self.right()
        left[:, self.slice] = right[:, self.slice]
        return left


class AverageLatents(MergeLatents):
    def __init__(
        self, left: Latents, right: Latents, depth: Union[MergeDepth, slice] = MergeDepth.ALL, left_weight: float = 0.5
    ):
        super().__init__(left, right, depth)
        self.left_weight = left_weight

    def forward(self):
        left, right = self.left(), self.right()
        left[:, self.slice] *= self.left_weight
        left[:, self.slice] += (1 - self.left_weight) * right[:, self.slice]
        return left


class ModulateLatents(MergeLatents):
    def __init__(self, left, right, feature, depth=MergeDepth.ALL, focus=None, smooth=2.0, causal=1.0, attenuate=1.0):
        super().__init__(left, right, depth)
        self.feature = feature
        self.focus = focus
        self.smooth = smooth
        self.causal = causal
        self.attenuate = attenuate

    def prepare(self, audio, sr, auxiliary=None):
        if self.focus is not None and auxiliary is not None:
            audio = auxiliary[self.focus]
        modulation = (
            normalize(gaussian_filter(self.feature(audio, sr), self.smooth, causal=self.causal)) * self.attenuate
        )
        self.register_buffer("modulation", modulation.reshape(modulation.shape[0], 1, 1))

    def forward(self):
        left, right = self.left(), self.right()
        left[:, self.slice] *= 1 - self.modulation
        left[:, self.slice] += self.modulation * right[:, self.slice]
        return left
