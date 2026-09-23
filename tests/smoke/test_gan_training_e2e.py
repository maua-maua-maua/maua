"""End-to-end GAN training smoke: a real LightningTrainer.fit() over the rewritten
(ffcv-free) data stack with the repo's actual components (deepconvolutional G/D,
normal latent space, softplus losses, blur augmentation)."""

import numpy as np
import PIL.Image
import pytest
import torch
import torchvision.transforms as tvt

pytestmark = [pytest.mark.gpu]


def test_gan_training_e2e(tmp_path):
    from pytorch_lightning import Trainer as LightningTrainer

    from maua.GAN.training.augmentation.blur import InitialBlur
    from maua.GAN.training.latent_spaces.normal import NormalLatentDistribution
    from maua.GAN.training.losses.softplus import DiscriminatorSoftPlus, GeneratorSoftPlus
    from maua.GAN.training.models.deepconvolutional import (
        DeepConvolutionalDiscriminator,
        DeepConvolutionalGenerator,
    )
    from maua.GAN.training.trainer import LightningGAN

    # tiny synthetic dataset
    data_dir = tmp_path / "imgs"
    data_dir.mkdir()
    rng = np.random.RandomState(0)
    for i in range(8):
        PIL.Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(data_dir / f"{i}.png")

    batch_size, z_dim, image_size = 4, 32, 64
    preprocess = tvt.Compose([tvt.Resize(image_size, antialias=True), tvt.CenterCrop(image_size)])
    pipeline = tvt.Compose(
        [tvt.PILToTensor(), tvt.ConvertImageDtype(torch.float32), tvt.Normalize([0.5] * 3, [0.5] * 3)]
    )

    gan = LightningGAN(
        latent=NormalLatentDistribution(batch_size=batch_size, z_dim=z_dim),
        generator=DeepConvolutionalGenerator(image_size=image_size, z_dim=z_dim, ngf=16),
        discriminator=DeepConvolutionalDiscriminator(image_size=image_size, ndf=16),
        discriminator_losses=[DiscriminatorSoftPlus()],
        generator_losses=[GeneratorSoftPlus()],
        shared_losses=[],
        augmentations=[InitialBlur(batch_size=batch_size, blur_init_sigma=0.0, blur_fade_kimg=1)],
        batch_size=batch_size,
        lr_G=2e-4,
        lr_D=2e-4,
        n_D_steps=1,
        input_dir=str(data_dir),
        preprocess=preprocess,
        pipeline=pipeline,
        cache_dir=str(tmp_path / "cache"),
        num_workers=0,
        jpeg_quality=95,
        epoch_kimg=1,
        test_kimg=10**9,  # never trigger the heavy metric pass in a smoke test
        monitor_metric="Frechet SwAV Distance",
    )

    trainer = LightningTrainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(gan)
    assert trainer.state.finished, "training loop must run to completion"
    assert trainer.global_step > 0, "optimizers must have stepped"

    # generator produces plausible images after fit
    out = gan.forward()
    assert out.shape[1:] == (3, image_size, image_size)
    assert torch.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 1
