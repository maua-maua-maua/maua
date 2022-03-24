#%%
import os
import random
from glob import glob
from math import ceil
from random import choice

import padl
import PIL.Image
import torch
from ffcv.fields import RGBImageField
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToDevice, ToTensor, ToTorchImage
from ffcv.writer import DatasetWriter
from IPython.display import clear_output, display
from torchvision import transforms as vision
from torchvision.transforms.functional import normalize
from tqdm import tqdm

vision = padl.transform(vision)

#%%
dataroot = "/home/hans/datasets/diffuse/diffuse/all/"
workers = 24
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5
ngpu = 1
ffcv_cache_path = "ds.beton"


#%%
"""Now we can compose any functions or callables with a nice piping syntax, combining transforms into a single pipeline. The pipeline has a handy print functionality, to really see what is going on in there."""


@padl.transform
def load_image(file):
    return PIL.Image.open(file).convert("RGB")


image_prep = (
    load_image
    >> vision.Resize(image_size)
    >> vision.CenterCrop(image_size)
    >> vision.ToTensor()
    >> padl.same.mul(255)
    >> padl.same.byte()
    >> padl.same.permute(1, 2, 0)
    >> padl.same.unsqueeze(0)
    >> padl.same.numpy()
)
image_prep


#%%
"""To check the intermediate steps of the padl.transform, we can use a handy subsetting functionality"""

file = choice(glob("/home/hans/datasets/diffuse/diffuse/all/*"))
item = image_prep(file)
print(item.min(), item.max(), item.shape, item.dtype)
item = normalize(torch.from_numpy(item).float().permute(0, 3, 1, 2), [127.5] * 3, [127.5] * 3)
print(item.min(), item.max(), item.shape, item.dtype)


#%%
"""We can define custom transforms by decorating functions or callable classes with `@padl.transform`. We can also wrap single functions as we do here with `PIL.Image.open`."""


images = [f"{dataroot}/{x}" for x in os.listdir(dataroot)]


class MyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(images)

    def __getitem__(self, idx):
        return image_prep(images[idx])


dataset = MyDataset()

if not os.path.exists(ffcv_cache_path):
    writer = DatasetWriter(ffcv_cache_path, {"image": RGBImageField(max_resolution=image_size, jpeg_quality=95)})
    writer.from_indexed_dataset(dataset)

loader = Loader(
    ffcv_cache_path,
    batch_size=batch_size,
    num_workers=workers,
    order=OrderOption.RANDOM,
    pipelines={"image": [SimpleRGBImageDecoder(), ToTensor(), ToTorchImage(), ToDevice(0)]},
)


def infiniter(loader):
    while True:
        for (batch,) in loader:
            yield batch


loader = infiniter(loader)


@padl.transform
def next_batch(*args, **kwargs):
    return next(loader)


#%%
"""Pytorch layers are first class citizens in PADL, and can be converted to PADL just as before with `@padl.transform`. PADL tracks all torch functionality by composing the class with a PADL object. In the wrapped class, PADL functionality is isolated under methods beginning `.pd_...`."""

import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


@padl.transform
class Generator(torch.nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
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
            torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)


@padl.transform
class Discriminator(torch.nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(
            # input is (nc) x 64 x 64
            torch.nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
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


netD = Discriminator(ngpu)
netG = Generator(ngpu)

#%%
"""We do something similar for the generator model.

Here we use the keyword `padl.same` which allows for a sort of neat inline lambda function. Standard `lambda` functions are also supported.

You'll also see the `padl.batch` and `padl.unbatch` keywords. These define where the preprocessing ends and forward pass begins, and forward pass ends and postprocessing begins.

When used in padl.batch-mode (see below), everything prior to the `padl.batch` is wrapped into a `torch.utils.data.DataLoader`. Every after `padl.unbatch` is mapped over the individual padl.batch elements of the forward pass. When used in single data-point mode, a single element padl.batch is constructed.

This leads to far less boilerplate, and far fewer errors with padl.batch dimensions, etc.. 

The *main* advantage of this, however, is that it allows the program to isolate all bits of code to run the generation pipeline, and to export these into a single portable saved artifact. This artifact may be then shared, compressed, imported into a serving environment etc..
"""


@padl.transform
def generate_noise(dummy):
    return torch.randn(nz, 1, 1)


@padl.transform
def denormalize(x):
    rescaled = 255 * (x * 0.5 + 0.5)
    converted = rescaled.numpy()
    return converted.astype(numpy.uint8)


generator = (
    generate_noise
    >> padl.batch
    >> netG
    >> padl.unbatch
    >> denormalize
    >> padl.same.transpose(1, 2, 0)
    >> padl.transform(PIL.Image.fromarray)
)
generator

#%%
"""Let's check the PADL-saved output. The saved artifact consists of a small python module, which includes only the bits of code which went into defining the generator. The saver tracks down all global variables, imports, functions, weights and data artifacts necessary for redefining and restoring the pipeline in its entirety. This is all packaged together into a compact, exportable directory."""

padl.save(generator, "test.padl", force_overwrite=True, compress=True)

#%%
"""When the keywords `padl.batch` or `padl.unbatch` are used, it's no longer to use the `__call__` methods directly anymore. Instead, the pipeline must be "applied" in one of three modes "train", "eval", and "infer". That's because the pipeline needs to be told how to construct the padl.batch, and whether to include gradients, and functionality only needed in training.

The modes are accessed with three key methods: `train_apply`, `eval_apply`, and `infer_apply`. With `infer_apply`, 
a single data-point padl.batch is created at the `padl.batch` point of the padl.transform, and then these padl.batch dimensions are removed again by the `padl.unbatch` statement.

In `train_apply` and `eval_apply`, a data loader is constructed on the fly and the batches out of this data loader are passed throught the forward pass. The padl.batch is then split into single rows after the `padl.unbatch` statement, and the postprocessing is mapped over these rows. In `train_apply` gradients are activated; in the other modes there are no gradients.

Let's apply the generator. Since it is a sampler, we can just pass an empty tuple or list of empty tuples.
"""

generator.infer_apply(())

#%%
"""We can dissect the generating pipeline into preprocessing, forward pass, postprocessing. Let's have a look and 
validate that `generator.pd_preprocess >> generator.pd_forward >> generator.pd_postproces` is equivalent to `generator`.
"""
generator.pd_preprocess
#%%
generator.pd_forward
#%%
generator.pd_postprocess
#%%
"""There are ways to create branches in the workflow using the operators `/`, `+` and `~`. See [here](link_to_the other_notebook) for details.
In the following part, we use `+` to add a label to the discriminator pipeline:
"""


@padl.transform
def real_label(x):
    return torch.ones_like(x)


criterion = padl.transform(torch.nn.BCELoss())


errD_real = (
    next_batch
    >> padl.same.float()
    >> vision.Normalize([127.5] * 3, [127.5] * 3)
    >> netD
    >> (padl.identity + real_label)
    >> criterion
)
errD_real
#%%


@padl.transform
def fake_label(x):
    return torch.zeros_like(x)


make_fake_tensor = generator.pd_preprocess >> generator.pd_forward


errD_fake = padl.same.detach() >> netD >> padl.identity + fake_label >> criterion
errD_fake

#%%
"""A test:"""

errD_fake.infer_apply(torch.randn(1, 3, 64, 64))

#%%
"""The generator pipeline:"""

errG = netD >> padl.identity + real_label >> criterion
errG

#%%
"""We can now create the optimizers and the iterators so that we can do some learning steps. Beware that
PyTorch requires specifying how the seed is set in each worker using `init_worker_fn` -- otherwise it's
possible to identical lines in the batches.
"""


def random_seed_init(i):
    torch.manual_seed(int(i))
    random.seed(int(i))
    numpy.random.seed(int(i))


optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

make_fake_tensor.pd_to("cuda")
errD_real.pd_to("cuda")
generator.pd_to("cuda")
errD_fake.pd_to("cuda")
errG.pd_to("cuda")

fake_generator = iter(
    make_fake_tensor.train_apply(
        range(1_000_000), batch_size=batch_size, num_workers=workers, worker_init_fn=random_seed_init
    )
)
errD_real_generator = iter(
    errD_real.train_apply(range(1_000_000), batch_size=batch_size, num_workers=workers, worker_init_fn=random_seed_init)
)

"""The training loop based on these pipelines is now super simple and (hopefully) sheds light on the important structure of how the DC-gan algorithm works.
"""
with tqdm(range(ceil(1_000_000 / batch_size)), unit_scale=batch_size, unit="img") as pbar:
    for it in pbar:

        fake_tensor = next(fake_generator)

        netD.zero_grad()
        ed_r = next(errD_real_generator)
        ed_r.backward()

        ed_f = errD_fake(fake_tensor)
        ed_f.backward()

        optimizerD.step()

        netG.zero_grad()
        eg = errG(fake_tensor)
        eg.backward()

        optimizerG.step()

        if it % 100 == 0:
            clear_output(wait=True)
            for j in range(5):
                display(generator.infer_apply())
            pbar.write(f"Iteration: {it}; ErrD/real: {ed_r:.3f}; ErrD/fake: {ed_f:.3f}; ErrG: {eg:.3f};")

#%%
"""Now let's padl.save the trained model!"""

padl.save(generator, "finished.padl")

#%%
"""A really useful feature, and making the finished pipeline super portable, is the ability to reload the full saved pipeline, without any importing or extra definitions. The following cell works, even after restarting the kernel/ or in a new session."""


reloader = padl.load("finished.padl")

#%%
"""We can now try a few sample generations from the trained pipeline, to check we get what we expect."""

reloader.infer_apply()
