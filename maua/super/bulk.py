import gc
import os
from glob import glob
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from tqdm import tqdm

from maua.super.waifu import merge, split


class Images(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filepath = self.images[index]
        return filepath.split("/")[-1], to_tensor(Image.open(filepath).convert("RGB"))


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{port}"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def writer(rank, world_size, q, out_dir):
    setup(rank, world_size, port=12346)

    filename, img = q.get()
    while filename is not None:
        name = f"{out_dir}/{Path(filename).stem}.png"
        save_image(img, name)
        del filename, img
        gc.collect()
        torch.cuda.empty_cache()
        filename, img = q.get()

    cleanup()


def worker(rank, world_size, q, dataset, scale, seg_size, pad_size, model_name, batch_size, jit=True):
    with torch.no_grad():
        setup(rank, world_size, port=12345)

        model = DistributedDataParallel(
            load_model(model_name=model_name, device=torch.device(f"cuda:{rank}")), device_ids=[rank]
        )
        dataloader = DataLoader(
            dataset=dataset,
            sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False),
            pin_memory=True,
        )

        def upscale(img: torch.Tensor, scale: int, pad_size: int, seg_size: int, batch_size: int):
            for _ in range(round(torch.log2(scale))):
                img_patches, h, w = split(img, 2, pad_size, seg_size)
                larger_patches = torch.cat([model(patches) for patches in torch.split(img_patches, batch_size)])
                img = merge(larger_patches, h, w, 2, pad_size, seg_size).clamp(0, 1)
            return img

        if jit:
            upscale = torch.jit.script(upscale, (next(iter(dataloader))[1], scale, pad_size, seg_size, batch_size))

        if rank == 0:
            dataloader = tqdm(dataloader)
        for (filename,), img in dataloader:
            img = upscale(img.cuda(rank).half(), rank, scale, pad_size, seg_size, batch_size)
            q.put((filename, img.to("cpu", non_blocking=True)))
        q.put((None, None))

        cleanup()


def bulk_main(args):
    seg_size = 64
    pad_size = 3
    world_size = torch.cuda.device_count()
    mp.set_start_method("spawn")

    images = sorted(glob(f"{args.input}/*"))
    dataset = Images(images)

    q = mp.Queue(maxsize=8)

    ctx = mp.spawn(
        worker,
        args=(world_size, q, dataset, args.scale, seg_size, pad_size, args.model_name, args.batch_size),
        nprocs=world_size,
        join=False,
        daemon=True,
    )

    num_writers = 1
    ctx2 = mp.spawn(writer, args=(num_writers, q, args.out_dir), nprocs=num_writers, join=True, daemon=True)
