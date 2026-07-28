"""DCGAN training script (v0).

Originally a PADL demo notebook (padl was dropped: abandoned upstream) using ffcv for
dataloading (also dropped: unmaintained, compile-heavy). Rewritten as plain
torch/torchvision with identical architecture, preprocessing, and training procedure.
"""

import os
import random
from glob import glob
from math import ceil

import numpy as np
import PIL.Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as vision
from tqdm import tqdm

# %% hyperparameters
dataroot = "/home/hans/datasets/diffuse/diffuse/all/"
workers = 24
batch_size = 128
image_size = 64
img_channels = 3
z_dim = 100
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5


image_prep = vision.Compose(
    [
        vision.Resize(image_size),
        vision.CenterCrop(image_size),
        vision.ToTensor(),
        vision.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.files = sorted(glob(f"{root}/*"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return image_prep(PIL.Image.open(self.files[idx]).convert("RGB"))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            torch.nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
            # state size. (img_channels) x 64 x 64
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            # input is (img_channels) x 64 x 64
            torch.nn.Conv2d(img_channels, ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            torch.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            torch.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            torch.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            torch.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)


def denormalize(x):
    rescaled = 255 * (x * 0.5 + 0.5)
    return rescaled.clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def generate(netG, device, n=1):
    z = torch.randn(n, z_dim, 1, 1, device=device)
    return [PIL.Image.fromarray(denormalize(img)) for img in netG(z)]


def random_seed_init(i):
    torch.manual_seed(int(i))
    random.seed(int(i))
    np.random.seed(int(i))


def infiniter(loader):
    while True:
        for batch in loader:
            yield batch


def main(total_images=1_000_000, out_dir="output/dcgan_v0"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(
        ImageFolderDataset(dataroot),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=random_seed_init,
    )
    batches = infiniter(loader)

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    criterion = torch.nn.BCELoss()

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    with tqdm(range(ceil(total_images / batch_size)), unit_scale=batch_size, unit="img") as pbar:
        for it in pbar:
            reals = next(batches).to(device)
            fakes = netG(torch.randn(len(reals), z_dim, 1, 1, device=device))

            # discriminator step
            netD.zero_grad()
            preds_real = netD(reals).view(-1)
            ed_r = criterion(preds_real, torch.ones_like(preds_real))
            ed_r.backward()
            preds_fake = netD(fakes.detach()).view(-1)
            ed_f = criterion(preds_fake, torch.zeros_like(preds_fake))
            ed_f.backward()
            optimizerD.step()

            # generator step
            netG.zero_grad()
            preds_fake = netD(fakes).view(-1)
            eg = criterion(preds_fake, torch.ones_like(preds_fake))
            eg.backward()
            optimizerG.step()

            if it % 100 == 0:
                for j, img in enumerate(generate(netG, device, n=5)):
                    img.save(f"{out_dir}/it{it:06d}_{j}.png")
                pbar.write(f"Iteration: {it}; ErrD/real: {ed_r:.3f}; ErrD/fake: {ed_f:.3f}; ErrG: {eg:.3f};")

    torch.save({"G": netG.state_dict(), "D": netD.state_dict()}, f"{out_dir}/finished.pt")


if __name__ == "__main__":
    main()
