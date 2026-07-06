import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]


def load_generator(model_file, architecture="stylegan2"):
    from maua.GAN.wrappers import get_generator_class

    G_cls = get_generator_class(architecture)
    return G_cls(model_file=str(model_file), output_size=(256, 256), strategy="stretch", layer=0).to("cuda")


def test_stylegan2_generate(stylegan2_model, assert_plausible_image):
    from maua.GAN.generate_images import generate_images

    def run():
        return list(
            generate_images(
                G=load_generator(stylegan2_model),
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

    imgs = run()
    assert len(imgs) == 1 and imgs[0].shape[-3] == 3
    assert_plausible_image(imgs[0].float())
    # the same seed must be reproducible
    imgs2 = run()
    assert torch.equal(imgs[0], imgs2[0]), "seeded generation is not deterministic"


def test_stylegan2_interpolation(stylegan2_model, assert_plausible_image):
    from maua.GAN.generate_interpolation import random_interpolation

    G = load_generator(stylegan2_model)
    frames = list(random_interpolation(G, n_frames=3, smooth=1, truncation=torch.tensor(1.0), batch_size=1))
    n_frames = sum(len(f) for f in frames)
    assert n_frames == 3, f"asked for 3 interpolation frames, got {n_frames}"
    for f in frames:
        assert_plausible_image(f.float())


@pytest.mark.slow
def test_projector(stylegan2_model, example_image, tmp_path):
    from maua.GAN.projector import project

    project(model_file=str(stylegan2_model), file=str(example_image), out_dir=str(tmp_path), steps=2)
    assert len(list(tmp_path.glob("*.jpg"))) > 0 or len(list(tmp_path.glob("*.pt"))) > 0


@pytest.mark.slow
def test_nada(stylegan2_model, tmp_path):
    from maua.GAN.nada import train

    # StyleGAN-NADA: CLIP-guided domain adaptation. Two iterations proves the ZSSGAN
    # forward/backward loop (which depends on the rescued maua.GAN.pix2pix submodule).
    train(
        checkpoint_path=str(stylegan2_model),
        source_class="photo",
        target_class="sketch",
        output_dir=str(tmp_path),
        size=256,
        batch=2,
        n_sample=8,
        iterations=1,
        output_interval=1,
        save_interval=1,
    )
    assert len(list(tmp_path.glob("*.jpg"))) > 0 or len(list(tmp_path.glob("*.pt"))) > 0
