import torch

from ... import audioreactive as ar
from ...patches.base.stylegan3 import StyleGAN3Patch


class ExampleSG3Patch(StyleGAN3Patch):
    def process_audio(self):
        vocals, drums, bass, other = ar.separate_sources(self.audio, self.sr, device=torch.device("cpu"))

        self.drum_onsets = ar.onsets(drums, self.sr, self.n_frames, margin=2, clip=95, smooth=2).reshape(-1, 1, 1)
        self.bass_rms = ar.rms(bass, self.sr, self.n_frames, smooth=20, clip=95, power=1).reshape(-1, 1, 1)
        self.vocal_rms = ar.rms(vocals, self.sr, self.n_frames, smooth=5, clip=95, power=1).reshape(-1, 1, 1)
        self.vocal_chroma = ar.chroma(vocals, self.sr, self.n_frames, margin=2)
        self.other_chroma = ar.chroma(other, self.sr, self.n_frames, margin=2)

        ar.plot_signals([self.drum_onsets, self.bass_rms, self.vocal_rms])
        ar.plot_spectra([self.vocal_chroma, self.other_chroma])

    def process_mapper_inputs(self):
        latent_z = self.stylegan3.get_z_latents("1-12,24-36,77-87,777-787,7777-7877")
        return {"latent_z": latent_z}

    def process_synthesizer_inputs(self, latent_w):
        vocal_chroma_latents = ar.chroma_weight_latents(self.vocal_chroma, latent_w[:12])
        other_chroma_latents = ar.chroma_weight_latents(self.other_chroma, latent_w[12:24])
        drum_latents = ar.spline_loops(latent_w[24:34], self.n_frames, n_loops=int(self.duration / 7))
        bass_latents = ar.spline_loops(latent_w[34:44], self.n_frames, n_loops=int(self.duration / 5))

        latent_w_plus = ar.spline_loops(latent_w[44:], self.n_frames, n_loops=1)
        latent_w_plus = (1 - self.vocal_rms) * latent_w_plus + self.vocal_rms * vocal_chroma_latents
        latent_w_plus[:, 10:] = other_chroma_latents[:, 10:]
        latent_w_plus = (1 - self.drum_onsets) * latent_w_plus + self.drum_onsets * drum_latents
        latent_w_plus = (1 - self.bass_rms) * latent_w_plus + self.bass_rms * bass_latents

        return {
            "latent_w_plus": latent_w_plus,
            "translation": torch.zeros((self.n_frames, 2)),
            "rotation": torch.zeros(self.n_frames),
        }
