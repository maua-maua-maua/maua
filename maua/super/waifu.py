import argparse
import gc
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision as tv
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append("maua/submodules/waifu2x")

from Models import CARN_V2, DCSCN, UpConv_7, Vgg_7, network_to_half


def load_model(which="upconv-anime", noise=1):
    if which == "upconv-anime":
        model = UpConv_7()
        model.load_pre_train_weights(
            json_file=f"maua/submodules/waifu2x/model_check_points/Upconv_7/anime/noise{noise}_scale2.0x_model.json"
        )
        model = network_to_half(model)
    elif which == "upconv-photo":
        model = UpConv_7()
        model.load_pre_train_weights(
            json_file=f"maua/submodules/waifu2x/model_check_points/Upconv_7/photo/noise{noise}_scale2.0x_model.json"
        )
        model = network_to_half(model)
    elif which == "vgg-ukbench":
        model = Vgg_7()
        model.load_pre_train_weights(
            json_file=f"maua/submodules/waifu2x/model_check_points/vgg_7/photo/noise{noise}_scale2.0x_model.json"
        )
        model = network_to_half(model)
    elif which == "vgg-art":
        model = Vgg_7()
        model.load_pre_train_weights(
            json_file=f"maua/submodules/waifu2x/model_check_points/vgg_7/photo/noise{noise}_scale2.0x_model.json"
        )
        model = network_to_half(model)
    elif which == "vgg-art-y":
        model = Vgg_7()
        model.load_pre_train_weights(
            json_file=f"maua/submodules/waifu2x/model_check_points/vgg_7/photo/noise{noise}_scale2.0x_model.json"
        )
        model = network_to_half(model)
    elif which == "vgg-photo":
        model = Vgg_7()
        model.load_pre_train_weights(
            json_file=f"maua/submodules/waifu2x/model_check_points/vgg_7/photo/noise{noise}_scale2.0x_model.json"
        )
        model = network_to_half(model)
    elif which == "carn":
        model = CARN_V2(
            color_channels=3,
            mid_channels=64,
            conv=nn.Conv2d,
            single_conv_size=3,
            single_conv_group=1,
            scale=2,
            activation=nn.LeakyReLU(0.1),
            SEBlock=True,
            repeat_blocks=3,
            atrous=(1, 1, 1),
        )
        model = network_to_half(model)
        model.load_state_dict(torch.load("maua/submodules/waifu2x/model_check_points/CRAN_V2/CARN_model_checkpoint.pt"))
    elif which == "dcscn":
        model = DCSCN(
            color_channel=3,
            up_scale=2,
            feature_layers=12,
            first_feature_filters=196,
            last_feature_filters=48,
            reconstruction_filters=64,
            up_sampler_filters=32,
        )
        model.load_state_dict("maua/submodules/waifu2x/model_check_points/DCSCN/DCSCN_weights_387epos_L12_noise_1.pt")
        model = network_to_half(model)
    return model.eval()


class Images(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filepath = self.images[index]
        return filepath.split("/")[-1], to_tensor(Image.open(filepath).convert("RGB"))


def windowed_index(device: torch.device, height: int, width: int, scale: int, pad_size: int, seg_size: int):
    if height % seg_size < pad_size or width % seg_size < pad_size:
        seg_size += scale * pad_size

    ys = torch.arange(pad_size, height, seg_size, dtype=torch.long, device=device)
    xs = torch.arange(pad_size, width, seg_size, dtype=torch.long, device=device)
    ys, xs = torch.meshgrid(ys, xs)
    idxs = torch.stack([ys.flatten(), xs.flatten()])

    winrange = torch.arange(-pad_size, pad_size + seg_size, dtype=torch.long, device=device)
    ywin, xwin = torch.meshgrid(winrange, winrange)
    window = torch.stack((ywin, xwin))

    idxs = idxs[:, :, None, None] + window[:, None, :, :]

    return idxs[0].clamp(0, height - 1), idxs[1].clamp(0, width - 1)


def split(img: torch.Tensor, scale: int, pad_size: int, seg_size: int):
    img = nn.functional.pad(img, [pad_size] * 4, mode="replicate")
    _, _, height, width = img.size()
    ys, xs = windowed_index(img.device, height, width, scale, pad_size, seg_size)
    patch_box = img[:, :, ys, xs].squeeze().permute(1, 0, 2, 3)
    return patch_box, height, width


def merge(img: torch.Tensor, height: int, width: int, scale: int, pad_size: int, seg_size: int):
    ys, xs = windowed_index(img.device, height * scale, width * scale, scale, pad_size * scale, seg_size * scale)
    rem = pad_size * 2
    img = img[..., rem:-rem, rem:-rem].permute(1, 0, 2, 3).unsqueeze(0).float()
    ys = ys[..., rem:-rem, rem:-rem]
    xs = xs[..., rem:-rem, rem:-rem]
    out = torch.zeros((1, 3, height * scale, width * scale), device=img.device)
    out[:, :, ys, xs] = img
    return out[..., rem:-rem, rem:-rem]


def setup(rank, world_size, port):
    print("setup")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{port}"
    print("init process group pls")
    print(rank, world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print("setup complete")


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


def worker(rank, world_size, q, dataset, scale, seg_size, pad_size, model_name, noise, batch_size, jit=False):
    with torch.no_grad():
        setup(rank, world_size, port=12345)

        print("get model")
        model = DistributedDataParallel(load_model(which=model_name, noise=noise).cuda(rank), device_ids=[rank])
        print("get dataloader")
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
            upscale = torch.jit.script(upscale, (next(iter(dataloader))[1], scale, pad_size, seg_size))

        if rank == 0:
            dataloader = tqdm(dataloader)
        print("upscaling")
        for (filename,), img in dataloader:
            img = upscale(img.cuda(rank).half(), rank, scale, pad_size, seg_size)
            q.put((filename, img.to("cpu", non_blocking=True)))
        q.put((None, None))

        cleanup()


def upscale(img, model_name="upconv-anime", noise=1, scale=2, pad_size=3, seg_size=64, batch_size=450):
    model = load_model(model_name, noise).to(img.device)
    for _ in range(round(np.log2(scale))):
        img_patches, h, w = split(img, 2, pad_size, seg_size)
        larger_patches = torch.cat([model(patches) for patches in torch.split(img_patches, batch_size)])
        img = merge(larger_patches, h, w, 2, pad_size, seg_size).clamp(0, 1)
    return img


def argument_parser():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--model_name", default="upconv-anime", choices=["upconv-anime", "upconv-photo", "vgg-ukbench", "vgg-art", "vgg-art-y", "vgg-photo", "carn", "dcscn"])
    parser.add_argument("--noise", default=1, choices=[0, 1, 2])
    parser.add_argument("--scale", default=2, choices=[2, 4, 8, 16])
    parser.add_argument("--out_dir", default="output/")
    # fmt: on
    return parser


def main(args):
    img = upscale(to_tensor(Image.open(args.input).convert("RGB")), args.model_name, args.noise, args.scale)
    save_image(img, f"{args.out_dir}/{Path(args.input).stem}.png")


def bulk_argument_parser():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--model_name", default="upconv-anime", choices=["upconv-anime", "upconv-photo", "vgg-ukbench", "vgg-art", "vgg-art-y", "vgg-photo", "carn", "dcscn"])
    parser.add_argument("--noise", default=1, choices=[0, 1, 2])
    parser.add_argument("--scale", default=2, choices=[2, 4, 8, 16])
    parser.add_argument("--batch_size", default=450, type=int)
    parser.add_argument("--out_dir", default="output/")
    # fmt: on
    return parser


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
        args=(world_size, q, dataset, args.scale, seg_size, pad_size, args.model_name, args.noise, args.batch_size),
        nprocs=world_size,
        join=False,
        daemon=True,
    )

    num_writers = 1
    ctx2 = mp.spawn(writer, args=(num_writers, q, args.out_dir), nprocs=num_writers, join=True, daemon=True)
    print("Done")
