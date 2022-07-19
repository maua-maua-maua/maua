import os

import numpy as np
import PIL
import torch
import torch.multiprocessing as mp
from ffcv.fields import RGBImageField
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter
from torch.utils.data import Dataset as TorchDataset


@torch.inference_mode()
def infiniter(loader):
    while True:
        for batch in loader:
            yield batch


class Iterator(object):
    def __init__(self, loader, kimg):
        self.loader = loader
        self.kimg = kimg
        self.count = 0
        self.endless = infiniter(self.loader)

    def __len__(self):
        return self.kimg * 1000 // self.loader.batch_size

    @torch.inference_mode()
    def __next__(self):
        if self.count >= self.kimg * 1000:
            raise StopIteration()
        (batch,) = next(self.endless)
        self.count += len(batch)
        return batch

    def __iter__(self):
        self.count = 0
        return self


def ImageLoader(
    files,
    ffcv_preprocess,
    ffcv_pipeline,
    cache_path,
    epoch_kimg=5,
    batch_size=16,
    num_workers=mp.cpu_count(),
    jpeg_quality=95,
) -> Loader:

    construct_loader = lambda: Loader(
        fname=cache_path,
        batch_size=batch_size,
        num_workers=num_workers,
        os_cache=True,
        order=OrderOption.QUASI_RANDOM,
        pipelines={"image": ffcv_pipeline},
    )

    class FFCVPreprocessorDataset(TorchDataset):
        def __len__(self):
            return len(files)

        def __getitem__(self, idx):
            return np.asarray(ffcv_preprocess(PIL.Image.open(files[idx]).convert("RGB")))[np.newaxis]

    data = FFCVPreprocessorDataset()
    size = max(data[0].shape)

    try:
        loader = construct_loader()
        (batch,) = next(iter(loader))
        assert max(batch.shape[-2:]) == size, "Preprocessed data does not match currently specified size!"
        rebuild = False
    except:
        if os.path.exists(cache_path):
            import traceback

            print("\nError while constructing dataloader, rebuilding dataset...")
            traceback.print_exc()
        rebuild = True

    if rebuild:
        DatasetWriter(
            cache_path, {"image": RGBImageField(max_resolution=size, jpeg_quality=jpeg_quality)}
        ).from_indexed_dataset(data)

        loader = construct_loader()
        print("Finished dataset preprocessing!\n")

    iterator = Iterator(loader, epoch_kimg)
    iterator.path = cache_path

    return iterator
