import torch
from maua.ops.image import random_cutouts
from resize_right import resize
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class MauaCutouts(nn.Module):
    def __init__(self, cut_size, cutn, pow_gain=16.0):
        super().__init__()
        self.cut_size, self.cutn, self.pow_gain = cut_size, cutn, pow_gain

    def forward(self, input, t):
        pow = self.pow_gain ** ((500 - t.float()) / 500)  # schedule to start large crops and end with small crops
        return random_cutouts(input, self.cut_size, self.cutn, pow)


class Cutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomPerspective(distortion_scale=0.4, p=0.7),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.15),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        )

    def forward(self, input, t):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(
                    max_size
                    * torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(float(self.cut_size / max_size), 1.0)
                )
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resize(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


class DangoCutouts(nn.Module):
    def __init__(
        self,
        cut_size,
        cut_overview=[12] * 400 + [4] * 600,
        cut_innercut=[4] * 400 + [12] * 600,
        cut_pow=1,
        cut_icgray_p=[0.2] * 400 + [0] * 600,
        animation_mode="Video Input",
        skip_augs=False,
    ):
        super().__init__()
        self.cut_size = cut_size
        self.cut_overview = cut_overview
        self.cut_innercut = cut_innercut
        self.cut_pow = cut_pow
        self.cut_icgray_p = cut_icgray_p
        self.skip_augs = skip_augs
        if animation_mode == "None":
            self.augs = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomGrayscale(p=0.1),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ]
            )
        elif animation_mode == "Video Input":
            self.augs = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomPerspective(distortion_scale=0.4, p=0.7),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomGrayscale(p=0.15),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ]
            )
        elif animation_mode == "2D" or animation_mode == "3D":
            self.augs = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.4),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomGrayscale(p=0.1),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
                ]
            )

    def forward(self, input, t):
        overview = self.cut_overview[999 - t]
        inner_crop = self.cut_innercut[999 - t]
        ic_grey_p = self.cut_icgray_p[999 - t]

        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(
            input,
            ((sideY - max_size) // 2, (sideY - max_size) // 2, (sideX - max_size) // 2, (sideX - max_size) // 2),
            mode="reflect",
        )
        cutout = resize(pad_input, out_shape=output_shape)

        if overview > 0:
            if overview <= 4:
                if overview >= 1:
                    cutouts.append(cutout)
                if overview >= 2:
                    cutouts.append(gray(cutout))
                if overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(overview):
                    cutouts.append(cutout)

        if inner_crop > 0:
            for i in range(inner_crop):
                size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(ic_grey_p * inner_crop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)

        cutouts = torch.cat(cutouts)
        if not self.skip_augs:
            cutouts = self.augs(cutouts)
        return cutouts


def make_cutouts(cutouts, cut_size, cutn, **cutout_kwargs):
    if cutouts == "normal":
        return Cutouts(cut_size, cutn, **cutout_kwargs)
    elif cutouts == "maua":
        return MauaCutouts(cut_size, cutn, **cutout_kwargs)
    elif cutouts == "dango":
        return DangoCutouts(cut_size, **cutout_kwargs)
    else:
        raise Exception(f"Cutouts {cutouts} not recognized!")
