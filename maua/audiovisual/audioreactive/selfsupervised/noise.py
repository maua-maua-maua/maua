import torch
from torch.nn.functional import pad

from maua.audiovisual.audioreactive.selfsupervised.features.audio import onsets
from maua.audiovisual.audioreactive.selfsupervised.features.processing import gaussian_filter, normalize
from maua.audiovisual.audioreactive.selfsupervised.latent import MergeDepth
from maua.audiovisual.audioreactive.selfsupervised.mir import estimate_tempo


def get_sizes(downscale_factor, aspect_ratio):
    assert downscale_factor <= 4
    size = 4
    sizes = [(round(size / downscale_factor), round(aspect_ratio * size / downscale_factor))]
    for _ in range(8):
        size *= 2
        sizes.append((round(size / downscale_factor), round(aspect_ratio * size / downscale_factor)))
        sizes.append((round(size / downscale_factor), round(aspect_ratio * size / downscale_factor)))
    return sizes


class Noise(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed if seed is not None else torch.randint(2**32 - 1, size=()).item()


class MultiplyNoise(Noise):
    def __init__(self, feature, seed=None):
        super().__init__(seed)
        self.feature = feature

    def prepare(self, audio, sr, auxiliary=None):
        self.register_buffer("modulation", normalize(gaussian_filter(self.feature(audio, sr), 2)))

    def forward(self, i, b, downscale_factor=1, aspect_ratio=1):
        modulation = self.modulation[i : i + b]
        modulation = modulation.reshape(len(modulation), -1)
        noise = []
        for s, (h, w) in enumerate(get_sizes(downscale_factor, aspect_ratio)):
            base_noise = torch.randn(
                (modulation.shape[1], h, w),
                device=modulation.device,
                generator=torch.Generator(modulation.device).manual_seed(self.seed + s),
            )
            noise.append(torch.einsum("MHW,BM->BHW", base_noise, modulation / modulation.sum(1, keepdim=True)))
        return noise


class LoopNoise(Noise):
    def __init__(self, sigma, n_loops=None, loop_bars=None, tempo=None, seed=None):
        super().__init__(seed)
        self.sigma = sigma

        assert n_loops is not None or loop_bars is not None
        self.n_loops = n_loops
        self.loop_bars = loop_bars

        self.tempo = tempo

    def prepare(self, audio, sr, auxiliary=None):
        if self.tempo is None:
            self.tempo = estimate_tempo(audio, sr)
        self.n_loops = self.n_loops or len(audio) / sr / 60 * self.tempo / 4 / self.loop_bars
        self.register_buffer("idx", torch.linspace(0, self.n_loops * 2 * torch.pi, len(onsets(audio, sr))))

    def forward(self, i, b, downscale_factor=1, aspect_ratio=1):
        noise = []
        for s, (h, w) in enumerate(get_sizes(downscale_factor, aspect_ratio)):
            base_noise = torch.randn(
                (3, h, w), device=self.idx.device, generator=torch.Generator(self.idx.device).manual_seed(self.seed + s)
            )

            freqs = torch.cos(self.idx[i : i + b, None, None] + base_noise[[0]]).div(self.sigma / 50)
            out = torch.sin(freqs + base_noise[[1]]) * base_noise[[2]]
            out = out / (out.square().mean(dim=(1, 2), keepdim=True).sqrt() + torch.finfo(out.dtype).eps)

            noise.append(out)
        return noise


class ConstantNoise(Noise):
    def __init__(self, noise_file) -> None:
        super().__init__(None)
        self.noise_file = noise_file

    def prepare(self, audio, sr, auxiliary=None):
        if self.noise_file is not None:
            self.noise_idxs = []
            for i, noise in enumerate(torch.load(self.noise_file)["noise"]):
                self.noise_idxs.append(i)
                self.register_buffer(f"noises{i}", noise)

    def forward(self, i, b, downscale_factor=1, aspect_ratio=1):
        noises = [self.get_buffer(f"noises{i}").squeeze(0) for i in self.noise_idxs]
        _, h, w = noises[0].shape
        if w / h != aspect_ratio:
            for n, noise in enumerate(noises):
                _, h, w = noise.shape
                aspect_shortage = aspect_ratio - w / h
                padding = round(aspect_shortage * h / 2)
                noises[n] = pad(noise, (padding, padding), mode="reflect")
        return noises


class MergeNoise(torch.nn.Module):
    def __init__(self, left, right, depth: MergeDepth | slice = MergeDepth.ALL):
        super().__init__()
        self.left = left
        self.right = right
        self.slice = depth.value

    def prepare(self, audio, sr, auxiliary=None):
        pass


class OverwriteNoise(MergeNoise):
    def forward(self, i, b, downscale_factor=1, aspect_ratio=1):
        left, right = self.left(i, b, downscale_factor, aspect_ratio), self.right(i, b, downscale_factor, aspect_ratio)
        for l, r in zip(left[self.slice], right[self.slice]):
            l.set_(r)
        return left


class AverageNoise(MergeNoise):
    def __init__(self, left, right, depth: MergeDepth | slice = MergeDepth.ALL, left_weight: float = 0.5):
        super().__init__(left, right, depth)
        self.left_weight = left_weight

    def forward(self, i, b, downscale_factor=1, aspect_ratio=1):
        left, right = self.left(i, b, downscale_factor, aspect_ratio), self.right(i, b, downscale_factor, aspect_ratio)
        for l, r in zip(left[self.slice], right[self.slice]):
            l.set_(l * self.left_weight + r * (1 - self.left_weight))
        return left


class ModulateNoise(MergeNoise):
    def __init__(
        self,
        left,
        right,
        feature,
        depth: MergeDepth | slice = MergeDepth.ALL,
        focus: str = None,
        smooth: float = 2,
    ):
        super().__init__(left, right, depth)
        self.feature = feature
        self.focus = focus
        self.smooth = smooth

    def prepare(self, audio, sr, auxiliary=None):
        if self.focus is not None:
            audio = auxiliary[self.focus]
        modulation = normalize(gaussian_filter(self.feature(audio, sr), self.smooth))
        self.register_buffer("modulation", normalize(modulation.reshape(len(modulation), -1).mean(1)))

    def forward(self, i, b, downscale_factor=1, aspect_ratio=1):
        modulation = self.modulation[i : i + b, None, None]
        left, right = self.left(i, b, downscale_factor, aspect_ratio), self.right(i, b, downscale_factor, aspect_ratio)
        for l, r in zip(left[self.slice], right[self.slice]):
            l.set_(l * (1 - modulation) + r * modulation)
        return left
