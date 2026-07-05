from pathlib import Path

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
def test_min_dalle():
    from maua.autoregressive.min_dalle.generate import generate

    images = generate(prompt="a red square", num_candidates=1, top_k=256, top_p=None, device="cuda")
    assert len(images) >= 1


@pytest.mark.slow
def test_ru_dalle(tmp_path):
    # importing this module puts the vendored `rudalle` submodule on sys.path, so import it first
    from maua.autoregressive.ru_dalle.generate import generate

    from rudalle import get_rudalle_model

    model = get_rudalle_model("Malevich", pretrained=True, fp16=True, device="cuda", cache_dir="modelzoo/")
    images = generate(
        model,
        input_text="a red square",
        num_outputs=1,
        batch_size=1,
        height=256,
        width=256,
        top_p=0.99,
        oversample=False,
        output_dir=str(tmp_path),
    )
    assert len(images) >= 1


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("modelzoo/rqvae_cc3m_cc12m_yfcc").exists(),
    reason="rqvae_cc3m_cc12m_yfcc checkpoint (multi-GB) not in local modelzoo; enable when present",
)
def test_rq_dalle(tmp_path):
    from maua.autoregressive.rq_dalle import main

    images = main(
        text_prompts="a red square",
        num_samples=1,
        sampling_ratio=1.0,
        batch_size=1,
        out_dir=str(tmp_path),
    )
    assert images is None or len(images) >= 1


def test_flux2hd():
    # imperative xfail: importing flux2hd downloads the full FLUX pipeline and hits hardcoded /home/hans paths
    pytest.xfail("flux2hd.py is script-style (import-time pipeline load, hardcoded paths); wrapped + moved to maua/text2img/flux.py in Phase C")
