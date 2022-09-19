import json

import numpy as np
import torch

from .latent import latent_patch, spline_loop_latents
from .mir import ALLFEATS, UNITFEATS
from .noise import Loop, noise_patch


def random_choice(rng, options, weights=None, n=1, replacement=False):
    if weights is None:
        probabilities = torch.ones(len(options), device=rng.device) / len(options)
    else:
        probabilities = torch.tensor(weights, device=rng.device) / np.sum(weights)

    idx = probabilities.multinomial(num_samples=n, replacement=replacement, generator=rng)

    return options[idx]


def skewnorm(rng, a, loc, scale, size=()):
    """
    skewnorm.pdf(x, a) = 2 * norm.pdf(x) * norm.cdf(a*x)
    From https://github.com/scipy/scipy/blob/main/scipy/stats/_continuous_distns.py#L7608-L7682
    """
    u0 = torch.randn(size, generator=rng, device=rng.device)
    v = torch.randn(size, generator=rng, device=rng.device)
    d = a / np.sqrt(1 + a**2)
    u1 = d * u0 + v * np.sqrt(1 - d**2)
    return loc + scale * torch.where(u0 >= 0, u1, -u1)


class Patch(torch.nn.Module):
    def __init__(
        self, features, segmentations, tempo, fps=24, seed=42, min_subpatches=2, max_subpatches=20, device="cuda"
    ):
        super().__init__()

        rng = torch.Generator(device)
        rng.manual_seed(seed)
        self.seed = seed
        self.rng = rng

        self.fps = fps
        self.tempo = tempo
        self.length = features[list(features.keys())[0]].shape[0]

        self.features = features
        self.segmentations = segmentations

        self.n_base_latents = torch.randint(3, 15, size=(), generator=rng, device=rng.device).item()
        self.sigma_base_noise = 1 + 9 * torch.rand((), generator=rng, device=rng.device).item()
        self.loops_base_noise = random_choice(rng, [1, 2, 4, 8, 16, 32, 64])

        self.ks = np.unique([k for (_, k) in segmentations]).tolist()

        self.min_subpatches, self.max_subpatches = min_subpatches, max_subpatches
        self.randomize_latent_patches()
        self.randomize_noise_patches()

    def __getstate__(self):
        # torch.Generator can't be pickled
        return {k: v for (k, v) in (list(self.__dict__.items()) + [("device", self.rng.device)]) if k != "rng"}

    def __setstate__(self, d):
        self.__dict__ = d
        self.rng = torch.Generator(d["device"]).manual_seed(d["seed"])  # ensure torch.Generator is re-initialized

    def randomize_latent_patches(self):
        self.latent_patches = [
            self.random_latent_patch()
            for _ in range(
                torch.randint(
                    self.min_subpatches, self.max_subpatches, size=(), generator=self.rng, device=self.rng.device
                )
            )
        ]

    def randomize_noise_patches(self):
        self.noise_patches = [
            self.random_noise_patch()
            for _ in range(
                torch.randint(
                    self.min_subpatches, self.max_subpatches, size=(), generator=self.rng, device=self.rng.device
                )
            )
        ]

    def update_intensity(self, val):
        for p in range(len(self.latent_patches)):
            self.latent_patches[p]["seq_feat_weight"] = skewnorm(self.rng, a=5, loc=val, scale=0.5).item()
            self.latent_patches[p]["mod_feat_weight"] = skewnorm(self.rng, a=5, loc=val, scale=0.5).item()
        for p in range(len(self.noise_patches)):
            self.noise_patches[p]["seq_feat_weight"] = skewnorm(self.rng, a=5, loc=val, scale=0.5).item()
            self.noise_patches[p]["mod_feat_weight"] = skewnorm(self.rng, a=5, loc=val, scale=0.5).item()
            self.latent_patches[p]["noise_std"] = skewnorm(self.rng, a=5, loc=val, scale=0.5).item()

    def random_latent_patch(self):
        return dict(
            patch_type=random_choice(self.rng, ["segmentation", "feature", "loop"]),
            segments=random_choice(self.rng, self.ks),
            loop_bars=random_choice(self.rng, [4, 8, 16, 32], weights=[2, 2, 2, 1]),
            seq_feat=random_choice(self.rng, ALLFEATS),
            seq_feat_weight=skewnorm(self.rng, a=5, loc=0.666, scale=0.5).item(),
            mod_feat=random_choice(self.rng, UNITFEATS),
            mod_feat_weight=skewnorm(self.rng, a=5, loc=0.666, scale=0.5).item(),
            merge_type=random_choice(self.rng, ["average", "modulate"], weights=[1, 3]),
            merge_depth=random_choice(
                self.rng, ["low", "mid", "high", "lowmid", "midhigh", "all"], weights=[3, 3, 3, 2, 2, 1]
            ),
        )

    def random_noise_patch(self):
        return dict(
            patch_type=random_choice(self.rng, ["blend", "multiply", "loop"]),
            loop_bars=random_choice(self.rng, [4, 8, 16, 32], weights=[2, 2, 2, 1]),
            seq_feat=random_choice(self.rng, ALLFEATS),
            seq_feat_weight=skewnorm(self.rng, a=5, loc=0.666, scale=0.5).item(),
            mod_feat=random_choice(self.rng, UNITFEATS),
            mod_feat_weight=skewnorm(self.rng, a=5, loc=0.666, scale=0.5).item(),
            merge_type=random_choice(self.rng, ["average", "modulate"], weights=[1, 3]),
            merge_depth=random_choice(
                self.rng, ["low", "mid", "high", "lowmid", "midhigh", "all"], weights=[3, 3, 3, 2, 2, 1]
            ),
            noise_mean=torch.randn((), generator=self.rng, device=self.rng.device).item() * 0.5,
            noise_std=skewnorm(self.rng, a=5, loc=0.666, scale=0.5).item(),
        )

    def forward(self, latent_palette, downscale_factor=1, aspect_ratio=1):
        self.rng.manual_seed(self.seed)

        base_selection = torch.randperm(len(latent_palette), generator=self.rng, device=self.rng.device)[
            : self.n_base_latents
        ]
        latents = spline_loop_latents(latent_palette[base_selection], self.length)
        for subpatch in self.latent_patches:
            latents = latent_patch(
                self.rng, latents, latent_palette, self.segmentations, self.features, self.tempo, self.fps, **subpatch
            )

        noise = [
            Loop(
                rng=self.rng,
                length=self.length,
                size=(round(aspect_ratio * size / downscale_factor), round(size / downscale_factor)),
                n_loops=self.loops_base_noise,
                sigma=self.sigma_base_noise,
            )
            for size in [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]
        ]
        for subpatch in self.noise_patches:
            noise = noise_patch(self.rng, noise, self.features, self.tempo, self.fps, **subpatch)

        return latents.to(self.rng.device), [n.to(self.rng.device) for n in noise]

    def __repr__(self):
        reprs = []
        for patches in [self.latent_patches, self.noise_patches]:
            header = [""] + [k for k in patches[0]]
            values = [
                [str(i + 1)]
                + [(f"{v:.4f}" if isinstance(v, float) else f"{v}").replace("spectral_", "") for v in p.values()]
                for i, p in enumerate(patches)
            ]
            widths = [max([len(row[n]) for row in [header] + values]) for n in range(len(header))]
            seps = ["-" * w for w in widths]
            strs = [" | ".join([row[c].ljust(widths[c]) for c in range(len(row))]) for row in [header, seps] + values]
            reprs.append(strs)
        return (
            "Patch(\n  Latent(\n    "
            + "\n    ".join(reprs[0])
            + "\n  ),\n  Noise(\n    "
            + "\n    ".join(reprs[1])
            + "\n  )\n)"
        )

    def save(self, path):
        with open(path, mode="w") as f:
            state = dict(
                seed=self.seed,
                latent_patches=self.latent_patches,
                noise_patches=self.noise_patches,
                n_base_latents=self.n_base_latents,
                sigma_base_noise=self.sigma_base_noise,
                loops_base_noise=self.loops_base_noise,
            )
            f.write(json.dumps(state))

    @staticmethod
    def load(path, features, segmentations, tempo, fps, device):
        patch = Patch(features, segmentations, tempo, fps, device)
        with open(path, mode="r") as f:
            patch_info = json.loads(f.read())
        for key, val in patch_info.items():
            setattr(patch, key, val)
        return patch
