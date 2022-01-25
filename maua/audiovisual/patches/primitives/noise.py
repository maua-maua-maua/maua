import torch
from ...audioreactive import gaussian_filter


class LoopNoise(torch.nn.Module):
    def __init__(self, loop_len, size, smooth):
        super().__init__()
        self.register_buffer("noise", gaussian_filter(torch.randn((loop_len, 1, size, size)), smooth))
        self.noise /= gaussian_filter(self.noise.std((1, 2, 3)), smooth).reshape(-1, 1, 1, 1)
        self.index, self.length = 0, loop_len

    def forward(self):
        noise = self.noise[[self.index % self.length]]
        self.index += 1
        return noise


class TempoLoopNoise(LoopNoise):
    def __init__(self, tempo, n_bars, fps, **loop_noise_kwargs):
        loop_len = n_bars * fps * 60 / (tempo / 4)
        super().__init__(loop_len, **loop_noise_kwargs)


class TonalNoise(torch.nn.Module):
    def __init__(self, chroma_or_tonnetz, size):
        super().__init__()
        chroma_or_tonnetz /= chroma_or_tonnetz.sum(0)
        noises = torch.randn(chroma_or_tonnetz.shape[0], 1, 1, size, size)
        self.register_buffer("noise", torch.sum(chroma_or_tonnetz[..., None, None, None] * noises, dim=0))
        self.index = 0

    def forward(self):
        noise = self.noise[[self.index]]
        self.index += 1
        return noise


class ModulatedNoise(torch.nn.Module):
    def __init__(self, modulation, base_noise):
        super().__init__()
        self.modulation = modulation
        self.base_noise = base_noise
        self.index = 0

    def forward(self):
        noise = self.modulation[self.index] * self.base_noise.forward()
        self.index += 1
        return noise


class CosSinNoise(torch.nn.Modules):
    def __init__(self, n_frames):
        pass
