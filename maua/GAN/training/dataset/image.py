import os

import numpy as np
import PIL.Image
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

# Preprocessed images are cached as individual JPEGs in a directory next to the requested
# cache path (formerly an ffcv .beton file — ffcv was dropped: unmaintained, compile-heavy).


@torch.inference_mode()
def infiniter(loader):
    while True:
        for batch in loader:
            yield batch


class Iterator:
    def __init__(self, loader, kimg):
        self.loader = loader
        self.kimg = kimg
        self.count = 0
        self.endless = infiniter(self.loader)
        self.path = None

    def __len__(self):
        return self.kimg * 1000 // self.loader.batch_size

    @torch.inference_mode()
    def __next__(self):
        if self.count >= self.kimg * 1000:
            raise StopIteration()
        batch = next(self.endless)
        self.count += len(batch)
        return batch

    def __iter__(self):
        self.count = 0
        return self


class CachedImageDataset(TorchDataset):
    """Applies `preprocess` once per image and caches the result as JPEG on disk.

    Subsequent epochs (and runs) read straight from the cache. `pipeline` is applied
    on-the-fly per sample (decode-time augmentation, analogous to the old ffcv pipeline).
    """

    def __init__(self, files, preprocess, pipeline, cache_dir, jpeg_quality=95):
        self.files = files
        self.preprocess = preprocess
        self.pipeline = pipeline
        self.cache_dir = cache_dir
        self.jpeg_quality = jpeg_quality
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.files)

    def cache_file(self, idx):
        return os.path.join(self.cache_dir, f"{idx:08d}.jpg")

    def __getitem__(self, idx):
        cached = self.cache_file(idx)
        if not os.path.exists(cached):
            img = self.preprocess(PIL.Image.open(self.files[idx]).convert("RGB"))
            if not isinstance(img, PIL.Image.Image):
                arr = np.asarray(img)
                img = PIL.Image.fromarray(arr.squeeze().astype(np.uint8))
            img.save(cached, quality=self.jpeg_quality)
        # always decode from the cache so every epoch sees identical (jpeg) data
        out = PIL.Image.open(cached).convert("RGB")
        if self.pipeline is not None:
            out = self.pipeline(out)
        return out


def ImageLoader(
    files,
    preprocess,
    pipeline,
    cache_path,
    epoch_kimg=5,
    batch_size=16,
    num_workers=mp.cpu_count(),
    jpeg_quality=95,
):
    # keep old .beton cache paths working by deriving a directory from them
    cache_dir = cache_path[:-len(".beton")] if cache_path.endswith(".beton") else cache_path
    dataset = CachedImageDataset(files, preprocess, pipeline, cache_dir, jpeg_quality)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    iterator = Iterator(loader, epoch_kimg)
    iterator.path = cache_dir

    return iterator
