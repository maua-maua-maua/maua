import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]

BACKENDS = [
    pytest.param("stable", marks=pytest.mark.backend_stable),
    pytest.param("guided", marks=pytest.mark.backend_guided),
    pytest.param("latent", marks=pytest.mark.backend_latent),
    pytest.param("glide", marks=pytest.mark.backend_glide),
    pytest.param("glid3xl", marks=pytest.mark.backend_glid3xl),
]


@pytest.mark.parametrize("diffusion", BACKENDS)
def test_text2img(diffusion, tiny):
    from maua.diffusion.image import image_sample

    img = image_sample(
        text="a colorful painting of a forest",
        sizes=[(tiny["size"], tiny["size"])],
        skips=[0.0],
        timesteps=tiny["steps"],
        diffusion=diffusion,
        sampler="plms",
    )
    assert torch.is_tensor(img) and img.ndim == 4 and img.shape[1] == 3


@pytest.mark.slow
@pytest.mark.xfail(reason="minDALL-E submodule uses mutable dataclass defaults, rejected by py>=3.12; see DEPRECATIONS.md", strict=False)
def test_min_dalle():
    from maua.autoregressive.min_dalle.generate import generate

    images = generate(prompt="a red square", num_candidates=1, top_k=256, top_p=None, device="cuda")
    assert len(images) >= 1


@pytest.mark.slow
@pytest.mark.skip(reason="multi-GB weights; enable during Phase B triage")
def test_ru_dalle():
    pass


@pytest.mark.slow
@pytest.mark.skip(reason="multi-GB weights; enable during Phase B triage")
def test_rq_dalle():
    pass


def test_flux2hd():
    # imperative xfail: importing flux2hd downloads the full FLUX pipeline and hits hardcoded /home/hans paths
    pytest.xfail("flux2hd.py is script-style (import-time pipeline load, hardcoded paths); repair in Phase B")
