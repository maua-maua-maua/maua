from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import splev, splrep

from ...audioreactive.inputs import slerp
from ...audioreactive.postprocess import gaussian_filter


class LoopLatents(torch.nn.Module):
    def __init__(self, latent_selection, loop_len, type="spline", smooth=10):
        super().__init__()

        latent_selection = latent_selection.cpu()

        if loop_len == 1 or type == "constant":
            latents = latent_selection[[0]]  # constant latent vector!
            loop_len = 1

        elif type == "spline":
            latent_selection = np.concatenate([latent_selection, latent_selection[[0]]])
            x = np.linspace(0, 1, loop_len)
            latents = np.zeros((loop_len, *latent_selection.shape[1:]))
            for lay in range(latent_selection.shape[1]):
                for lat in range(latent_selection.shape[2]):
                    tck = splrep(np.linspace(0, 1, latent_selection.shape[0]), latent_selection[:, lay, lat])
                    latents[:, lay, lat] = splev(x, tck)
            latents = torch.from_numpy(latents)

        elif type == "slerp":
            latent_selection = np.concatenate([latent_selection, latent_selection[[0]]])
            latents = []
            for n in range(len(latent_selection)):
                for val in np.linspace(0.0, 1.0, int(ceil(loop_len / len(latent_selection)))):
                    latents.append(
                        torch.from_numpy(
                            slerp(
                                latent_selection[n % len(latent_selection)][0],
                                latent_selection[(n + 1) % len(latent_selection)][0],
                                val,
                            )
                        )
                    )
            latents = torch.stack(latents).unsqueeze(1).tile(1, 18, 1)
            latents = F.interpolate(latents.permute(2, 1, 0), loop_len, mode="linear", align_corners=False).permute(
                2, 1, 0
            )
            latents = gaussian_filter(latents, 1)

        elif type == "gaussian":
            latents = torch.cat([lat.tile(round(loop_len / len(latent_selection)), 1, 1) for lat in latent_selection])
            latents = F.interpolate(latents.permute(2, 1, 0), loop_len, mode="linear", align_corners=False).permute(
                2, 1, 0
            )
            latents = gaussian_filter(latents, smooth)

        self.register_buffer("latents", latents)
        self.index, self.length = 0, loop_len

    def forward(self):
        latents = self.latents[[self.index % self.length]]
        self.index += 1
        return latents


class TempoLoopLatents(LoopLatents):
    def __init__(self, tempo, latent_selection, n_bars, fps, **loop_latents_kwargs):
        if len(latent_selection) == 1:
            loop_len = 1
        else:
            loop_len = round(n_bars * fps * 60 / (tempo / 4))
        super().__init__(latent_selection, loop_len, **loop_latents_kwargs)


class PitchTrackLatents(torch.nn.Module):
    def __init__(self, pitch_track, latent_selection):
        super().__init__()

        low, high = np.percentile(pitch_track, 25), np.percentile(pitch_track, 75)
        pitch_track -= low
        pitch_track /= high
        pitch_track *= len(latent_selection)
        print(pitch_track.min(), pitch_track.mean(), pitch_track.max())
        pitch_track = pitch_track.round().long().numpy()
        pitch_track = pitch_track % len(latent_selection)

        latents = torch.from_numpy(latent_selection.numpy()[pitch_track])
        self.register_buffer("latents", latents)
        self.index = 0

    def forward(self):
        latent = self.latents[[self.index]]
        self.index += 1
        return latent


class TonalLatents(torch.nn.Module):
    def __init__(self, chroma_or_tonnetz, latent_selection):
        super().__init__()
        chroma_or_tonnetz /= chroma_or_tonnetz.sum(1)[:, None]
        latents = torch.einsum(
            "Atwl,Atwl->twl",
            chroma_or_tonnetz.permute(1, 0)[..., None, None],
            latent_selection[: chroma_or_tonnetz.shape[1], None],
        )
        self.register_buffer("latents", latents)
        self.index = 0

    def forward(self):
        latents = self.latents[[self.index]]
        self.index += 1
        return latents


class ModulatedLatents(torch.nn.Module):
    def __init__(self, modulation, base_latents):
        super().__init__()
        self.register_buffer("latents", modulation[:, None, None] * base_latents[[0]])
        self.index = 0

    def forward(self):
        latents = self.latents[[self.index]]
        self.index += 1
        return latents


class LucidSonicDreamLatents(torch.nn.Module):
    pass


class LazyModulatedLatents(torch.nn.Module):
    pass
    # def __init__(self, modulation, base_latents):
    #     super().__init__()
    #     self.modulation = modulation
    #     self.base_latents = base_latents
    #     self.index = 0

    # def forward(self):
    #     latents = self.modulation[self.index] * self.base_latents.forward()
    #     self.index += 1
    #     return latents
