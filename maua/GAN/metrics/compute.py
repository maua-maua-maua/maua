import os
from glob import glob
from pathlib import Path
from typing import Callable, List, Union
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from .extractors import get_extractor
from .frechet import frechet_distance
from .kernel import kernel_distance
from .prdc import prdc

EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp", "npy"}
SIZE = 224


def resize_single_channel(x_np, size):
    img = Image.fromarray(x_np.astype(np.float32), mode="F")
    img = img.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(img).clip(0, 255).reshape(size, size, 1)


def resize_clean(x, size):
    x = [resize_single_channel(x[:, :, idx], size) for idx in range(3)]
    x = np.concatenate(x, axis=2).astype(np.float32) / 255.0
    return x


def resize(x, size):
    dev = x.device
    image_np = x.permute(0, 2, 3, 1).cpu().numpy()
    resized = torch.stack([to_tensor(resize_clean(im, size)) for im in image_np])
    return resized.to(dev)


class FolderImages(Dataset):
    def __init__(self, input_dir, n_images, size) -> None:
        super().__init__()

        if ".zip" in input_dir:
            files = list(set(ZipFile(input_dir).namelist()))
            files = [x for x in files if os.path.splitext(x)[1].lower()[1:] in EXTENSIONS]
        else:
            files = sorted(
                [file for ext in EXTENSIONS for file in glob(os.path.join(input_dir, f"**/*.{ext}"), recursive=True)]
            )

        files = np.array(files)
        if n_images < len(files):
            files = files[np.random.permutation(len(files))[:n_images]]

        self.files = files
        self.n_images = n_images
        self.input_dir = input_dir
        self.size = size

    def __getitem__(self, index: int) -> None:
        path = self.files[index]

        if ".zip" in self.input_dir:
            with ZipFile(self.input_dir).open(path, "r") as f:
                image_np = np.array(Image.open(f).convert("RGB"))
        if ".npy" in path:
            image_np = np.load(path)
        else:
            img_pil = Image.open(path).convert("RGB")
            image_np = np.array(img_pil)

        resized = resize_clean(image_np, self.size)

        return to_tensor(resized)

    def __len__(self) -> int:
        return self.n_images


class GeneratorImages(Dataset):
    def __init__(self, G, n_images, size) -> None:
        super().__init__()
        self.n_images = n_images
        self.G = G
        self.size = size

    def __getitem__(self, index: int) -> None:
        tensor = self.G().add(1).div(2)
        resized = resize(tensor, self.size)
        return resized

    def __len__(self) -> int:
        return self.n_images


@torch.inference_mode()
def compute(
    real_samples: Union[str, Path, DataLoader],
    fake_samples: Callable,
    n_samples: int = 10_000,
    extractor: str = "SwAV",
    metrics: List[str] = ["frechet", "kernel", "prdc"],
    batch_size: int = 32,
    num_workers: int = torch.multiprocessing.cpu_count(),
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ignore_cache: bool = False,
    verbose: bool = True,
):
    extractor_model, size = get_extractor(extractor)
    extractor_model = extractor_model.eval().to(device)

    if isinstance(real_samples, (str, Path)):
        cache_file = f"cache/{Path(real_samples).stem}_real_{extractor}_features.npz"
    else:
        cache_file = f"cache/{Path(real_samples.path).stem}.npz".replace("ffcv", f"real_{extractor}_features")
    use_cache = os.path.exists(cache_file) and not ignore_cache

    if isinstance(real_samples, (str, Path)):
        if not use_cache:
            real_loader = DataLoader(
                FolderImages(real_samples, n_samples, size),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            real_loader = range(n_samples // batch_size)
    else:
        real_loader = real_samples

    fake_loader = DataLoader(GeneratorImages(fake_samples, n_samples // batch_size, size), batch_size=1, num_workers=0)

    real_features, fake_features = [], []
    if verbose:
        pbar = tqdm(zip(real_loader, fake_loader), total=n_samples // batch_size, unit_scale=batch_size, unit="images")
    else:
        pbar = zip(real_loader, fake_loader)
    for real_batch, fake_batch in pbar:
        if not use_cache:
            real_batch = real_batch.to(device)
            fake_batch = fake_batch.squeeze().to(device)
            if real_batch.shape[-2:] != fake_batch.shape[-2:]:
                real_batch = resize(real_batch, size)
            assert len(real_batch) == len(fake_batch)
            batch = torch.cat((real_batch, fake_batch))
        else:
            batch = fake_batch.squeeze().to(device)

        feats = extractor_model(batch).cpu()

        if not use_cache:
            real_feats, fake_feats = feats.chunk(2)
            real_features.append(real_feats)
            fake_features.append(fake_feats)
        else:
            fake_features.append(feats)
    del pbar

    fake_features = torch.cat(fake_features)

    if not os.path.exists(cache_file):
        real_features = torch.cat(real_features)
        np.savez_compressed(cache_file, real_features=real_features.numpy())
    else:
        with np.load(cache_file) as data:
            real_features = torch.from_numpy(data["real_features"])

    metrics_dict = {}
    for metric in metrics:
        if metric == "frechet":
            metrics_dict[f"Frechet {extractor} Distance"] = frechet_distance(real_features, fake_features)
        elif metric == "kernel":
            metrics_dict[f"Kernel {extractor} Distance"] = kernel_distance(real_features, fake_features)
        elif metric == "prdc":
            (
                metrics_dict[f"Precision ({extractor})"],
                metrics_dict[f"Recall ({extractor})"],
                metrics_dict[f"Density ({extractor})"],
                metrics_dict[f"Coverage ({extractor})"],
            ) = prdc(real_features, fake_features)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metrics_dict


if __name__ == "__main__":
    import argparse
    import json

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--extractor", type=str, default="SwAV", choices=["SwAV", "Inception"])
    parser.add_argument("--metrics", type=str, nargs="+", default=["frechet", "kernel", "prdc"], choices=["frechet", "kernel", "prdc"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=torch.multiprocessing.cpu_count())
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ignore_cache", action="store_true")
    args = parser.parse_args()
    # fmt: on

    from maua.GAN.load import load_network

    G = load_network(args.checkpoint).eval().to(args.device)
    fake_samples = lambda *a, **kw: G(z=torch.randn((args.batch_size, 512), device=args.device), c=None).squeeze()

    metrics_dict = compute(
        real_samples=args.data_dir,
        fake_samples=fake_samples,
        n_samples=args.n_samples,
        extractor=args.extractor,
        metrics=args.metrics,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        ignore_cache=args.ignore_cache,
    )

    print(json.dumps(metrics_dict, indent=4))
