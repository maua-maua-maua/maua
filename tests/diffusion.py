import gc

import numpy as np
import pytest
import torch

from maua.diffusion.sample import main

DIFFUSION_SPEEDS = [
    ("latent", "n/a"),
    ("glid3xl", "n/a"),
    ("glide", "n/a"),
    ("guided", "fast"),
    ("guided", "regular"),
]
DIFFUSION_IDS = [f"{tup[0]}-{tup[1]}".replace("-n/a", "") for tup in DIFFUSION_SPEEDS]


@pytest.fixture(scope="module", params=DIFFUSION_SPEEDS, ids=DIFFUSION_IDS)
def diffusion_model(request):
    diffusion, speed = request.param

    sampler = "plms"
    timesteps = 5
    clip_scale = 2500
    cfg_scale = 5
    if diffusion == "guided":
        from maua.diffusion.processors.guided import GuidedDiffusion
        from maua.grad import CLIPGrads

        diffusion_model = GuidedDiffusion(
            [CLIPGrads(scale=clip_scale)], sampler=sampler, timesteps=timesteps, speed=speed
        )
    elif diffusion == "latent":
        from maua.diffusion.processors.latent import LatentDiffusion

        diffusion_model = LatentDiffusion(cfg_scale=cfg_scale, sampler=sampler, timesteps=timesteps)
    elif diffusion == "glide":
        from maua.diffusion.processors.glide import GLIDE

        diffusion_model = GLIDE(cfg_scale=cfg_scale, sampler=sampler, timesteps=timesteps)
    elif diffusion == "glid3xl":
        from maua.diffusion.processors.glid3xl import GLID3XL

        diffusion_model = GLID3XL(cfg_scale=cfg_scale, sampler=sampler, timesteps=timesteps)

    yield diffusion_model

    del diffusion_model
    gc.collect()
    torch.cuda.empty_cache()


SIZES = [
    [(128, 128)],
    [(256, 256)],
    [(128, 384)],
    [(384, 128)],
    [(128, 128), (256, 256), (384, 384)],
]
SIZE_ID = lambda x: "->".join(["x".join([str(sz) for sz in tup]) for tup in x])


@pytest.mark.parametrize("sizes", SIZES, ids=SIZE_ID)
def test_various_sizes(diffusion_model, sizes):
    img = main(
        text="beautiful rainforest leaves with silver veins, digital art",
        diffusion=diffusion_model,
        sizes=sizes,
        skips=np.linspace(0, 0.7, len(sizes)),
        stitch=False,
    )
    assert tuple(img.shape[-2:]) == sizes[-1]


TILE_SIZES = [128, 256, 384]


@pytest.mark.parametrize("tile_size", TILE_SIZES, ids=lambda x: f"tile{x}")
def test_stitching(diffusion_model, tile_size):
    img = main(
        text="beautiful rainforest leaves with silver veins, digital art",
        diffusion=diffusion_model,
        sizes=[(256, 256)],
        skips=[0],
        stitch=True,
        tile_size=tile_size,
    )
    assert tuple(img.shape[-2:]) == (256, 256)


SAMPLERS = ["p", "ddim", "plms"]


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_samplers(diffusion_model, sampler):
    main(
        text="beautiful rainforest leaves with silver veins, digital art",
        diffusion=diffusion_model,
        sizes=[(256, 256)],
        skips=[0],
        sampler=sampler,
    )
