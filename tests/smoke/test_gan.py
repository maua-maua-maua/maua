import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]


def load_generator(model_file, architecture="stylegan2"):
    from maua.GAN.wrappers import get_generator_class

    G_cls = get_generator_class(architecture)
    return G_cls(model_file=str(model_file), output_size=(256, 256), strategy="stretch", layer=0).to("cuda")


def test_stylegan2_generate(stylegan2_model):
    from maua.GAN.generate_images import generate_images

    G = load_generator(stylegan2_model)
    imgs = list(
        generate_images(
            G=G,
            seeds=[42],
            class_idx=None,
            truncation=1.0,
            latent_sampling="standard",
            langevin_critic=None,
            translation=None,
            rotation=None,
            batch_size=1,
        )
    )
    assert len(imgs) == 1 and imgs[0].shape[-3] == 3


def test_stylegan2_interpolation(stylegan2_model):
    from maua.GAN.generate_interpolation import random_interpolation

    G = load_generator(stylegan2_model)
    frames = list(random_interpolation(G, n_frames=3, smooth=1, truncation=torch.tensor(1.0), batch_size=1))
    assert len(frames) >= 1


@pytest.mark.slow
def test_projector():
    pytest.xfail("projector.py has hardcoded /home/hans paths; repair in Phase B")


@pytest.mark.slow
@pytest.mark.skip(reason="NADA/blending/SeFa need per-method triage in Phase B")
def test_nada_blend_sefa():
    pass
