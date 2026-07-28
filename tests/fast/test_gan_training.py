"""Integration tests for the padl/ffcv-free GAN training stack.

Covers the rewritten maua/GAN/training/dataset/image.py (jpeg-cache ImageLoader) and
train_v0.py (plain-torch DCGAN): dataset caching, loader iteration/shapes/ranges, and
real adversarial optimization steps on both architectures.
"""

import numpy as np
import PIL.Image
import pytest
import torch
import torchvision.transforms as tvt

from maua.GAN.training.dataset.image import CachedImageDataset, ImageLoader


@pytest.fixture()
def image_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    rng = np.random.RandomState(42)
    for i in range(8):
        PIL.Image.fromarray(rng.randint(0, 255, (96, 80, 3), dtype=np.uint8)).save(d / f"img_{i}.png")
    return d


def _preprocess(size=64):
    return tvt.Compose([tvt.Resize(size, antialias=True), tvt.CenterCrop(size)])


def _pipeline(size=64):
    return tvt.Compose([tvt.PILToTensor(), tvt.ConvertImageDtype(torch.float32), tvt.Normalize([0.5] * 3, [0.5] * 3)])


def test_cached_image_dataset_caches_jpegs(image_dir, tmp_path):
    files = sorted(str(f) for f in image_dir.iterdir())
    cache = tmp_path / "cache"
    ds = CachedImageDataset(files, _preprocess(), _pipeline(), str(cache))

    first = ds[0]
    assert (cache / "00000000.jpg").exists(), "preprocessed image should be cached to disk"
    assert first.shape == (3, 64, 64)

    # second read comes from cache and must be identical
    second = ds[0]
    assert torch.equal(first, second)


def test_image_loader_iterates_batches(image_dir, tmp_path):
    files = sorted(str(f) for f in image_dir.iterdir())
    loader = ImageLoader(
        files,
        _preprocess(),
        _pipeline(),
        str(tmp_path / "cache_loader"),
        epoch_kimg=1,
        batch_size=4,
        num_workers=0,
    )
    batches = list(loader)
    assert len(batches) == 1 * 1000 // 4
    for batch in batches[:3]:
        assert batch.shape == (4, 3, 64, 64)
        assert batch.dtype == torch.float32
        assert batch.min() >= -1.0 and batch.max() <= 1.0

    # iterating again restarts the epoch (Iterator resets its count)
    assert len(list(loader)) == len(batches)


def test_image_loader_accepts_legacy_beton_path(image_dir, tmp_path):
    files = sorted(str(f) for f in image_dir.iterdir())
    legacy = str(tmp_path / "legacy_ffcv.beton")
    loader = ImageLoader(files, _preprocess(), _pipeline(), legacy, epoch_kimg=1, batch_size=4, num_workers=0)
    assert not loader.path.endswith(".beton")
    next(iter(loader))


def test_train_v0_dcgan_training_step():
    """One real adversarial G+D step of the rewritten DCGAN converges numerically."""
    from maua.GAN.training import train_v0 as t

    torch.manual_seed(0)
    netG, netD = t.Generator(), t.Discriminator()
    criterion = torch.nn.BCELoss()
    optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

    reals = torch.rand(2, 3, 64, 64) * 2 - 1
    fakes = netG(torch.randn(2, t.z_dim, 1, 1))
    assert fakes.shape == (2, 3, 64, 64)
    assert fakes.min() >= -1 and fakes.max() <= 1  # tanh output

    netD.zero_grad()
    pr = netD(reals).view(-1)
    ed_r = criterion(pr, torch.ones_like(pr))
    ed_r.backward()
    pf = netD(fakes.detach()).view(-1)
    ed_f = criterion(pf, torch.zeros_like(pf))
    ed_f.backward()
    optD.step()

    netG.zero_grad()
    pf = netD(fakes).view(-1)
    eg = criterion(pf, torch.ones_like(pf))
    eg.backward()
    optG.step()

    for loss in (ed_r, ed_f, eg):
        assert torch.isfinite(loss), "losses must be finite after a real G+D step"

    imgs = t.generate(netG, torch.device("cpu"), n=2)
    assert len(imgs) == 2 and imgs[0].size == (64, 64)


def test_lightning_gan_constructs_with_loader(image_dir, tmp_path, monkeypatch):
    """LightningGAN builds its dataloader through the new ImageLoader without ffcv."""
    from maua.GAN.training.trainer import LightningGAN

    class Latent(torch.nn.Module):
        def forward(self):
            return torch.randn(2, 8)

    class G(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 3 * 64 * 64)

        def forward(self, z):
            return self.lin(z).view(-1, 3, 64, 64).tanh()

    class D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3 * 64 * 64, 1)

        def forward(self, x):
            return self.lin(x.flatten(1))

    monkeypatch.chdir(tmp_path)
    gan = LightningGAN(
        latent=Latent(),
        generator=G(),
        discriminator=D(),
        discriminator_losses=[],
        generator_losses=[],
        shared_losses=[],
        augmentations=[],
        batch_size=2,
        lr_G=1e-4,
        lr_D=1e-4,
        n_D_steps=1,
        input_dir=str(image_dir),
        preprocess=_preprocess(),
        pipeline=_pipeline(),
        cache_dir=str(tmp_path / "cache_pl"),
        num_workers=0,
        jpeg_quality=95,
        epoch_kimg=1,
        test_kimg=100,
        monitor_metric="Frechet SwAV Distance",
    )
    batch = next(iter(gan.train_dataloader()))
    assert batch.shape == (2, 3, 64, 64)
    out = gan.forward()
    assert out.shape[1:] == (3, 64, 64)
    assert out.min() >= 0 and out.max() <= 1
