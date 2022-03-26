from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import padl
import PIL
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
import torchvision as tv
from ffcv.fields import RGBImageField
from ffcv.fields.decoders import SimpleRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToDevice, ToTensor, ToTorchImage
from ffcv.writer import DatasetWriter
from padl.transforms import Parallel as PadlParallel
from padl.transforms import Transform as PadlTransform
from padl_ext.pytorch_lightning.prepare import LightningModule as PadlLightningModule
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import Module as TorchModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms as tv_transforms
from torchvision.transforms.functional import InterpolationMode

from .dataset.image import ImageLoader

tv_transforms = padl.transform(tv_transforms)


def ImageLoader(
    padl_dataloader,
    ffcv_preprocess,
    ffcv_pipeline,
    preprocess_image_size,
    cache_path,
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

    try:
        loader = construct_loader()
        rebuild = False
    except:
        rebuild = True

    if rebuild:
        padl_dataset = padl_dataloader.dataset

        class FFCVPreprocessor(TorchDataset):
            def __len__(self):
                return len(padl_dataset)

            def __getitem__(self, idx):
                return ffcv_preprocess(padl_dataset[idx])

        DatasetWriter(
            cache_path, {"image": RGBImageField(max_resolution=preprocess_image_size, jpeg_quality=jpeg_quality)}
        ).from_indexed_dataset(
            FFCVPreprocessor(),
        )

        loader = construct_loader()

    return loader


class GAN(PadlLightningModule):
    def __init__(
        self,
        # modules
        generate_latent: TorchModule,
        generator: TorchModule,
        discriminator: TorchModule,
        discriminator_losses: List[TorchModule],
        generator_losses: List[TorchModule],
        common_losses: List[TorchModule],
        preprocess: PadlTransform,
        postprocess: PadlTransform,
        # settings
        lr_G: float,
        lr_D: float,
        n_D_steps: int,
        cache_dir: str,
        input_dir: str,
        # lightning
        trainer: LightningTrainer,
        train_data: TorchDataset,
        val_data: TorchDataset = None,
        test_data: TorchDataset = None,
        **kwargs,
    ):
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.n_D_steps = n_D_steps
        self.data_cache_path = f"{cache_dir}/{Path(input_dir).stem}_ffcv.beton"

        @padl.transform
        def discriminate(fakes, reals, **kwargs):
            fake_logits = discriminator(fakes.detach())
            real_logits = discriminator(reals)
            return fake_logits, real_logits

        train_discriminator = (
            ((generate_latent >> generator) / preprocess) + discriminate + PadlParallel(discriminator_losses)
        )
        train_generator = generate_latent >> generator >> discriminator >> PadlParallel(generator_losses)
        train_model = (train_discriminator / train_generator) + PadlParallel(common_losses)
        inference_model = generate_latent >> padl.batch >> generator >> padl.unbatch >> postprocess

        super().__init__(
            padl_model=train_model,
            trainer=trainer,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            inference_model=inference_model,
        )
        del self.learning_rate
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        # fmt:off
        parser = parent_parser.add_argument_group("Optimizers")
        parser.add_argument("--lr_G", type=float, default=1e-4, help="Starting learning rate for the generator")
        parser.add_argument("--lr_D", type=float, default=1e-4, help="Starting learning rate for the discriminator")
        parser.add_argument("--n_D_steps", type=int, default=1, help="How many discriminator steps to perform per generator step")

        parser = parent_parser.add_argument_group("Dataloader")
        parser.add_argument("--cache_dir", type=str, default="cache/", help="Directory to cache preprocessed data in")
        parser.add_argument("--jpeg_quality", type=int, default=95, help="Quality of preprocessed jpegs stored in cache. Lower values will save disk space at the cost of image quality.")
        # fmt:on
        return parent_parser

    def forward(self):
        return self.inference_model.infer_apply()

    def train_dataloader(self):
        padl_loader = super().train_dataloader()
        if padl_loader is None:
            return None
        return ImageLoader(padl_loader)

    def val_dataloader(self):
        padl_loader = super().val_dataloader()
        if padl_loader is None:
            return None
        return ImageLoader(padl_loader)

    def test_dataloader(self):
        padl_loader = super().test_dataloader()
        if padl_loader is None:
            return None
        return ImageLoader(padl_loader)

    def training_step(self, batch, batch_idx, optimizer_idx):
        _, batch = batch
        return self.padl_model.pd_forward.pd_call_in_mode((batch, batch_idx, optimizer_idx), "train")

    def validation_step(self, batch, batch_idx):
        _, batch = batch
        return self.padl_model.pd_forward.pd_call_in_mode(batch, "eval")

    def test_step(self, batch, batch_idx):
        _, batch = batch
        return self.padl_model.pd_forward.pd_call_in_mode(batch, "eval")

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=(0, 0.99))
        scheduler_G = ReduceLROnPlateau(optimizer_G)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_D, betas=(0, 0.99))
        scheduler_D = ReduceLROnPlateau(optimizer_D)
        return (
            {"optimizer": optimizer_G, "lr_scheduler": scheduler_G},
            {
                "optimizer": optimizer_D,
                "lr_scheduler": {
                    "scheduler": scheduler_D,
                    "monitor": "frechet_ema",
                },
                "frequency": self.n_D_steps,
            },
        )


def main(args):
    dict_args = vars(args)

    if args.generator == "dcgan":
        generator = DeepConvolutionalGenerator(**dict_args)

    if args.discriminator == "dcgan":
        discriminator = DeepConvolutionalDiscriminator(**dict_args)

    if args.latent_distribution == "normal":
        generate_latent = NormalLatentDistribution(**dict_args)

    preprocess = (
        padl.transform(lambda file: PIL.Image.open(file).convert("RGB"))
        >> tv_transforms.Resize(args.preprocess_image_size)
        >> tv_transforms.CenterCrop(args.preprocess_image_size)
        >> tv_transforms.ToTensor()
    )

    ffcv_preprocess = lambda x: x.mul(255).byte().permute(1, 2, 0).unsqueeze(0).numpy()

    if args.random_crop:
        ffcv_decoder = RandomResizedCropRGBImageDecoder(
            args.image_size,
            scale=(1 / args.random_crop_zoom, 1),
            ratio=(1 - args.random_crop_ratio, 1 + args.random_crop_ratio),
        )
    else:
        ffcv_decoder = SimpleRGBImageDecoder()
    ffcv_pipeline = [ffcv_decoder, ToTensor(), ToTorchImage(), tv_transforms.Normalize([127.5] * 3, [127.5] * 3)]
    if args.hflip:
        ffcv_pipeline.append(tv_transforms.RandomHorizontalFlip())
    if args.vflip:
        ffcv_pipeline.append(tv_transforms.RandomVerticalFlip())
    if args.random_rotate:
        # calculate minimum padding needed to ensure no non-padded pixels end up in image
        # no padding needed at 0, sqrt(2)*image_radius for 45 degrees or more
        padding = np.ceil(
            np.sqrt(2) * args.image_size * (1 - np.cos(4 * np.pi * min(args.random_rotate_degrees, 45) / 180)) / 4
        )
        ffcv_pipeline += [
            torch.nn.ReflectionPad2d(padding),
            tv_transforms.RandomRotation(args.random_rotate_degrees, interpolation=InterpolationMode.BILINEAR),
            tv_transforms.Resize(args.image_size),
        ]

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            EarlyStopping(monitor="frechet_ema"),
            ModelCheckpoint(monitor="frechet_ema", every_n_epochs=1, save_top_k=5),
        ],
    )

    padl_lightning_module = GAN(
        generate_latent=generate_latent,
        generator=generator,
        discriminator=discriminator,
        discriminator_losses=discriminator_losses,
        generator_losses=generator_losses,
        common_losses=common_losses,
        preprocess=preprocess,
        postprocess=postprocess,
        lr_G=args.lr_G,
        lr_D=args.lr_D,
        n_D_steps=args.n_D_steps,
        cache_dir=args.cache_dir,
        input_dir=args.input_dir,
        trainer=trainer,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )
    padl_lightning_module.fit()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter())
    parser = GAN.add_argparse_args(parser)

    # fmt: off
    parser.add_argument("-G", "--generator", type=str, default="deepconvolutional", help="which generator to train with")
    parser.add_argument("-D", "--discriminator", type=str, default="deepconvolutional", help="which discriminator to train with")
    # fmt: on

    temp_args, _ = parser.parse_known_args()

    if temp_args.generator == "deepconvolutional":
        from .models.deepconvolutional import DeepConvolutionalGenerator

        parser = DeepConvolutionalGenerator.add_model_specific_args(parser)
    elif temp_args.model_name == "stylehypermixerfly":
        from .models.stylehypermixerfly import StyleHyperMixerFlyGenerator

        parser = StyleHyperMixerFlyGenerator.add_model_specific_args(parser)
    else:
        raise ValueError(f"Unknown generator {temp_args.generator}")

    if temp_args.discriminator == "deepconvolutional":
        from .models.deepconvolutional import DeepConvolutionalDiscriminator

        parser = DeepConvolutionalDiscriminator.add_model_specific_args(parser)
    elif temp_args.model_name == "stylehypermixerfly":
        from .models.stylehypermixerfly import HyperMixerFlyDiscriminator

        parser = HyperMixerFlyDiscriminator.add_model_specific_args(parser)
    else:
        raise ValueError(f"Unknown discriminator {temp_args.discriminator}")

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
