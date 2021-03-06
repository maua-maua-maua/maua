import torch

from ... import audioreactive as ar
from ...patches.base.stylegan2 import StyleGAN2Patch

# audio input | filtering | feature | GAN input  | merge |
# ----------- | --------- | ------- | ---------- | ----- |
# vocal       | low       | chroma  | latent     |       |
# drums       | high      | onsets  | latent     |       |
# bass        | not       | volume  | truncation |       |


class ExampleSG2Patch(StyleGAN2Patch):
    def process_audio(self):
        vocals, drums, bass, other = ar.separate_sources(self.audio, self.sr, device=torch.device("cpu"))

        self.kick_onsets = ar.onsets(
            ar.low_pass(drums, self.sr, 100, 24), self.sr, self.n_frames, margin=2, clip=95, smooth=2
        )
        self.snare_onsets = ar.onsets(
            ar.band_pass(drums, self.sr, 100, 400, 24), self.sr, self.n_frames, margin=2, clip=95, smooth=2
        )
        self.drum_onsets = ar.onsets(drums, self.sr, self.n_frames, margin=2, clip=95, smooth=2).reshape(-1, 1, 1)
        self.bass_rms = ar.rms(bass, self.sr, self.n_frames, smooth=20, clip=95, power=1).reshape(-1, 1, 1)
        self.vocal_rms = ar.rms(vocals, self.sr, self.n_frames, smooth=5, clip=95, power=1).reshape(-1, 1, 1)
        self.vocal_chroma = ar.chroma(vocals, self.sr, self.n_frames, margin=2)
        self.other_chroma = ar.chroma(other, self.sr, self.n_frames, margin=2)

        ar.plot_signals([self.drum_onsets, self.bass_rms, self.vocal_rms])
        ar.plot_spectra([self.vocal_chroma, self.other_chroma])

    def process_mapper_inputs(self):
        latent_z = self.stylegan2.get_z_latents("1-12,24-36,77-87,777-787,7777-7877")
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

        noise_slow = ar.gaussian_filter(torch.randn((self.n_frames, 1, 64, 64)), 15)
        noise_slow /= ar.gaussian_filter(noise_slow.std((1, 2, 3)), 5).reshape(-1, 1, 1, 1)
        noise_fast = ar.gaussian_filter(torch.randn((self.n_frames, 1, 64, 64)), 3)
        noise_fast /= 0.5 * ar.gaussian_filter(noise_fast.std((1, 2, 3)), 5).reshape(-1, 1, 1, 1)
        noise = (1 - self.drum_onsets[..., None]) * noise_slow + self.drum_onsets[..., None] * noise_fast
        noises = self.synthesizer.make_noise_pyramid(noise)
        noises["noise0"] = torch.tile(noises["noise0"][[self.n_frames // 2]], (self.n_frames, 1, 1, 1))
        noises["noise1"] = torch.tile(noises["noise1"][[self.n_frames // 2]], (self.n_frames, 1, 1, 1))
        noises["noise2"] = torch.tile(noises["noise2"][[self.n_frames // 2]], (self.n_frames, 1, 1, 1))

        translation = torch.cat((0.1 * (1 - self.snare_onsets.reshape(-1, 1)), torch.zeros(self.n_frames, 1)), dim=1)
        zoom = 1 - 0.3 * self.kick_onsets
        rotation = self.kick_onsets * 5 * ar.gaussian_filter(torch.randn((self.n_frames,)), 1)

        return {
            "latent_w_plus": latent_w_plus,
            "zoom": zoom,
            "translation": translation,
            "rotation": rotation,
            **noises,
        }
