import argparse
import random
from glob import glob
from math import ceil
from pathlib import Path
from uuid import uuid4

import torch
from numpy import sqrt
from PIL import Image
from torch.nn import ReflectionPad2d
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import RandomCrop, RandomRotation
from torchvision.transforms.functional import center_crop, to_pil_image, to_tensor
from tqdm import tqdm


def cover_crop(im):
    c, h, w = im.shape
    imgs = []
    if w > h:
        for x in torch.linspace(0, w - h, ceil(w / h), dtype=torch.long):
            imgs.append(im[:, :, x : x + h])
    else:
        for y in torch.linspace(0, h - w, ceil(h / w), dtype=torch.long):
            imgs.append(im[:, y : y + w, :])
    return torch.stack(imgs)


class MultiCropDataset(Dataset):
    def __init__(self, directory, cover=True, random=False, sizes=[1024], number=10, rotate=False) -> None:
        super().__init__()
        self.files = sum([glob(f"{directory}/*{ext}") for ext in IMG_EXTENSIONS], [])
        self.cover = cover
        self.random = random
        self.sizes = sizes
        self.number = number
        self.rotate = rotate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]

        try:
            im = Image.open(file).convert("RGB")
            L = min(im.size)
            im = to_tensor(im)

            crops = []

            if self.cover:
                crops.append(cover_crop(im))

            if self.random:
                for _ in range(self.number):
                    if self.rotate:
                        im = ReflectionPad2d(L - 1)(im)
                        im = RandomRotation(self.rotate)(im)
                        im = center_crop(im, round(L * sqrt(2)))
                    im = RandomCrop(random.choice(self.sizes), pad_if_needed=True, padding_mode="reflect")(im)
                    crops.append(im.unsqueeze(0))
        except Exception as e:
            print()
            print("ERROR", e)
            print(file)
            print()
            crops = []

        return file, crops


def collate_fn(batch):
    return batch[0][0], sum([[b.squeeze(0)] if b.shape[0] == 1 else list(b.unbind(0)) for b in batch[0][1]], [])


if __name__ == "__main__":
    # fmt:off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=True)
    parser.add_argument("in_dir", type=str, help="Directory containing image files")
    parser.add_argument("out_dir", type=str, help="Directory to save output files to")
    parser.add_argument("--extension", type=str, default="png", help="Extension of images to save")
    parser.add_argument("--random", action="store_true", help="Enable random cropping from images")
    parser.add_argument("--sizes", type=int, default=[1024], nargs="+", help="Sizes to crop at (chosen randomly per crop)")
    parser.add_argument("--number", type=int, default=10, help="Number of random crops")
    parser.add_argument("--rotate", type=int, default=0, help="Max angle of random rotations before crops (+/-)")
    parser.add_argument("--no-cover", action="store_true", help="Disable default square crops which span the entire image")
    parser.add_argument("--no-uuid", action="store_true", help="Disable prepending of unique strings to output file names")
    args = parser.parse_args()
    # fmt:on

    dataset = MultiCropDataset(
        directory=args.in_dir,
        cover=not args.no_cover,
        random=args.random,
        sizes=args.sizes,
        number=args.number,
        rotate=args.rotate,
    )
    dataloader = DataLoader(dataset, shuffle=True, num_workers=torch.multiprocessing.cpu_count(), collate_fn=collate_fn)

    def save(file_crops):
        file, crops = file_crops
        if len(crops) == 0:
            return
        for img in crops:
            out_name = f"{Path(args.in_dir).stem}_{Path(file).stem}"
            if not args.no_uuid:
                out_name = f"{str(uuid4())[:6]}_{out_name}"
            to_pil_image(img).save(f"{args.out_dir}/{out_name}.{args.extension}")

    with torch.multiprocessing.Pool(torch.multiprocessing.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(save, dataloader), total=len(dataloader)):
            pass
