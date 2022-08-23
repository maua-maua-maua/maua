import gc

import numpy as np
import pytest
import torch
from maua.diffusion.image import get_diffusion_model, image_sample
from maua.diffusion.video import video_sample

DIFFUSION_SPEEDS = [
    ("stable", "n/a"),
    ("latent", "n/a"),
    ("glid3xl", "n/a"),
    ("glide", "n/a"),
    ("guided", "fast"),
    ("guided", "regular"),
]
SIZES = [
    [(128, 128)],
    [(256, 256)],
    [(128, 384)],
    [(384, 128)],
    [(128, 128), (256, 256), (384, 384)],
]
TILE_SIZES = [128, 256, 384]
SAMPLERS = ["p", "ddim", "plms"]

DIFFUSION_IDS = [f"{tup[0]}-{tup[1]}".replace("-n/a", "") for tup in DIFFUSION_SPEEDS]
SIZE_ID = lambda x: "->".join(["x".join([str(sz) for sz in tup]) for tup in x])

# ======================================================================================================================
# ==================================================== SAMPLE TESTS ====================================================
# ======================================================================================================================


@pytest.fixture(scope="module", params=DIFFUSION_SPEEDS, ids=DIFFUSION_IDS)
def diffusion_model(request):
    diffusion, speed = request.param
    diffusion_model = get_diffusion_model(diffusion, timesteps=5, guidance_speed=speed)
    yield diffusion_model
    del diffusion_model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("sizes", SIZES, ids=SIZE_ID)
def test_various_sizes(diffusion_model, sizes):
    img = image_sample(
        text="beautiful rainforest leaves with silver veins, digital art",
        diffusion=diffusion_model,
        sizes=sizes,
        skips=np.linspace(0, 0.7, len(sizes)),
        stitch=False,
    )
    assert tuple(img.shape[-2:]) == sizes[-1]


@pytest.mark.parametrize("tile_size", TILE_SIZES, ids=lambda x: f"tile{x}")
def test_stitching(diffusion_model, tile_size):
    img = image_sample(
        text="beautiful rainforest leaves with silver veins, digital art",
        diffusion=diffusion_model,
        sizes=[(256, 256)],
        skips=[0],
        stitch=True,
        tile_size=tile_size,
    )
    assert tuple(img.shape[-2:]) == (256, 256)


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_samplers(diffusion_model, sampler):
    image_sample(
        text="beautiful rainforest leaves with silver veins, digital art",
        diffusion=diffusion_model,
        sizes=[(256, 256)],
        skips=[0],
        sampler=sampler,
    )


@pytest.mark.parametrize(
    "sampler", ["p", "ddim", "plms", "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms"]
)
def test_stable_samplers(sampler):
    image_sample(
        text="beautiful rainforest leaves with silver veins, digital art",
        diffusion="stable",
        sizes=[(512, 512)],
        skips=[0],
        sampler=sampler,
    )


# =====================================================================================================================
# ==================================================== VIDEO TESTS ====================================================
# =====================================================================================================================


def test_video_init(diffusion_model):
    video_sample(
        diffusion=diffusion_model,
        init="/home/hans/datasets/video/dreams24.mp4",
        text="beautiful rainforest leaves with silver veins, digital art",
        size=(256, 256),
        timesteps=5,
    )
