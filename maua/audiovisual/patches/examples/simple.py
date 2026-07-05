"""Minimal audio-reactive StyleGAN2 patch.

The smallest complete example of the patch contract: map a few random latents,
spline-interpolate them into a smooth loop over the whole clip, and modulate that
loop with the track's volume envelope so louder moments pull toward a fixed pose.

It is intentionally tiny (no source separation, no per-stem features) so it stays
readable and fast; the ``stylegan2.py`` / ``stylegan3.py`` patches in this folder are
richer but are written against an older ``audioreactive`` API and need porting.
"""

import torch

from maua.audiovisual import audioreactive as ar
from maua.audiovisual.patches.base.stylegan2 import StyleGAN2Patch


class SimpleSG2Patch(StyleGAN2Patch):
    def process_audio(self):
        # RMS envelope in [0, 1], one value per output frame.
        self.volume = ar.resample(ar.volume(self.audio, self.sr), self.n_frames).reshape(-1, 1, 1)

    def process_mapper_inputs(self):
        # A handful of keyframes to interpolate between.
        return {"latent_z": torch.randn((4, 512))}

    def process_synthesizer_inputs(self, latent_w):
        latents = ar.spline_loops(latent_w, self.n_frames, n_loops=1)
        # Pull toward the first keyframe on loud moments for an audio-reactive pulse.
        latents = (1 - 0.3 * self.volume) * latents + (0.3 * self.volume) * latent_w[[0]]
        return {"latents": latents}
