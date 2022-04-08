import importlib
import inspect
import os
from glob import glob
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tvt
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder
from ffcv.transforms import ToTensor, ToTorchImage
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from .trainer import LightningGAN, WeightsEMA

HERE = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":
    # fmt: off
    import argparse

    class IgnorantActionsContainer(argparse._ActionsContainer):
        def _handle_conflict_ignore(self, action, conflicting_actions):
            pass

    argparse.ArgumentParser.__bases__ = (argparse._AttributeHolder, IgnorantActionsContainer)
    argparse._ArgumentGroup.__bases__ = (IgnorantActionsContainer,)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        conflict_handler="ignore",
        allow_abbrev=True,
    )
    parser.add_argument("-lh", "--show_lightning_help", action="store_true", help="Show PyTorch Lightning Trainer arguments in --help message")
    parser.add_argument("-e", "--experimental", action="store_true", help="Enable experimental network options")

    subparser = parser.add_argument_group("Input data", description="Settings related to data to train with (augmentations applied here will be visible in output data)")
    subparser.add_argument("--input_dir", type=str, help="Directory containing image files to train on")
    subparser.add_argument("--image_size", type=int, default=256, help="Size of images to train on")
    subparser.add_argument("--preprocess_image_size", type=int, default=256, help="Size of images during preprocessing (if using random crops with zoom, you can set this to the same factor times your image_size to ensure crops never have insufficient pixel density")
    subparser.add_argument("--hflip", action="store_true", help="Whether to randomly flip images horizontally")
    subparser.add_argument("--vflip", action="store_true", help="Whether to randomly flip images vertically")
    subparser.add_argument("--random_crop", action="store_true", help="Apply random crops to data while training")
    subparser.add_argument("--random_crop_zoom", type=float, default=np.sqrt(2), help="Maximum amount random crops are allowed to zoom into images (e.g. 2 means 2x zoom or cropping into a section half the size of the original image)")
    subparser.add_argument("--random_crop_ratio", type=float, default=0.1, help="Maximum amount that image ratio may be changed by random cropping (e.g. 0.25 allows ratios between 3/4 and 4/3 which will result in slightly stretched images when resized back to square)")
    subparser.add_argument("--random_rotate", action="store_true", help="Whether to apply random rotations to input data")
    subparser.add_argument("--random_rotate_degrees", type=float, default=360, help="Maximum amount of degrees to rotate by")

    subparser = parser.add_argument_group("Dataloading", description="Settings related to caching/loading data during training")
    subparser.add_argument("--batch_size", type=int, default=32, help="Directory containing image files to train on")
    subparser.add_argument("--num_workers", type=int, default=8, help="Number of background thread workers loading images")
    subparser.add_argument("--jpeg_quality", type=int, default=95, help="Quality of preprocessed jpegs stored in cache. Lower values will save disk space at the cost of image quality.")
    subparser.add_argument("--cache_dir", type=str, default="cache/", help="Directory to cache preprocessed data in")

    subparser = parser.add_argument_group("Training", description="General training and logging settings")
    subparser.add_argument("--kimg", type=int, default=100_000, help="How many thousands of images to train on in total")
    subparser.add_argument("--epoch_kimg", type=int, default=10, help="How many kimg per epoch")
    subparser.add_argument("--ckpt_kimg", type=int, default=10, help="How many thousands of images to train on between model checkpoints")
    subparser.add_argument("--ckpt_top_k", type=int, default=10, help="How many checkpoints to keep")
    subparser.add_argument("--test_kimg", type=int, default=100, help="How many epochs to wait between running metric analysis")
    subparser.add_argument("--monitor_metric", type=str, default="Frechet SwAV Distance", help="Which metric to use to monitor progress for learning rate decay, top checkpoints, and early stopping.")

    subparser = parser.add_argument_group("Optimization", description="Settings related optimizers")
    subparser.add_argument("--lr_G", type=float, default=2e-4, help="Starting learning rate for the generator")
    subparser.add_argument("--lr_D", type=float, default=2e-4, help="Starting learning rate for the discriminator")
    subparser.add_argument("--n_D_steps", type=int, default=1, help="How many discriminator steps to perform per generator step")

    temp_args, _ = parser.parse_known_args()
    override_args = {}

    loss_choices = [Path(f).stem for f in glob(f"{HERE}/losses/*.py")]
    model_choices = [Path(f).stem for f in glob(f"{HERE}/models/*.py")]
    if temp_args.experimental:
        loss_choices.extend([Path(f).stem for f in glob(f"{HERE}/losses/experimental/*.py")])
        model_choices.extend([Path(f).stem for f in glob(f"{HERE}/models/experimental/*.py")])
    latent_choices = [Path(f).stem for f in glob(f"{HERE}/latent_spaces/*.py")]
    augment_choices = [Path(f).stem for f in glob(f"{HERE}/augmentation/*.py")]

    for component_module, component_title, component_desc in [
        ("models", "Model settings", "Settings of the various different supported models"),
        ("latent_spaces", "Latent space settings", "Settings related to different choices of latent space"),
        ("losses", "Loss settings", "Settings related to functions used to train the GAN"),
        ("augmentation", "Training augmentations", "Settings related to augmentations for improving training (augmentations applied here will NOT be visible in output data)"),
    ]:
        subparser = parser.add_argument_group(component_title, description=component_desc)
        if component_module == "models":
            subparser.add_argument("-L", "--latent_distribution", type=str, default="normal", choices=latent_choices, help="which input latent distribution to use")
            subparser.add_argument("-G", "--generator", type=str, default="deepconvolutional", choices=model_choices, help="which generator to use")
            subparser.add_argument("-D", "--discriminator", type=str, default="deepconvolutional", choices=model_choices, help="which discriminator to use")
            subparser.add_argument("-EMA", "--ema_decay", type=float, default=0.995, help="model weight exponential moving average decay")
        if component_module == "losses":
            subparser.add_argument("-DL", "--discriminator_losses", type=str, nargs="+", default=["softplus"], choices=loss_choices, help="which discriminator losses to use")
            subparser.add_argument("-GL", "--generator_losses", type=str, nargs="+", default=["softplus"], choices=loss_choices, help="which generator losses to use")
        if component_module == "augmentation":
            subparser.add_argument("-A", "--augmentations", type=str, nargs="+", default=["blur"], choices=augment_choices, help="which augmentations to use")

        files = glob(f"{HERE}/{component_module}/*.py")
        if temp_args.experimental:
            files.extend(glob(f"{HERE}/{component_module}/experimental/*.py"))
        for file in files:
            name = Path(file).stem
            try:
                mod = importlib.import_module(f".{component_module}.{name}", package=__package__)
            except:
                mod = importlib.import_module(f".{component_module}.experimental.{name}", package=__package__)
            for torch_module_name, torch_module in [
                (n, m) for n, m in inspect.getmembers(mod, inspect.isclass) if m.__module__ == mod.__name__
            ]:
                if hasattr(torch_module, "add_model_specific_args"):
                    parser = torch_module.add_model_specific_args(parser)
                if hasattr(torch_module, "override_args"):
                    override_args = {**override_args, **torch_module.override_args}
    # fmt: on

    if temp_args.show_lightning_help:
        parser = LightningTrainer.add_argparse_args(parser)
    parser.add_argument("-h", "--help", action="help")
    args, _ = parser.parse_known_args()
    if not args.show_lightning_help:
        parser = LightningTrainer.add_argparse_args(parser)
        args = parser.parse_args()

    for k, v in override_args.items():
        setattr(args, k, v)

    args_dict = vars(args)

    # =============================================================
    # ======================= BUILD MODULES =======================
    # =============================================================

    def build_by_name(module, submodule, pred=lambda m: True):
        try:
            mod = importlib.import_module(f".{module}.{submodule}", package=__package__)
        except:
            mod = importlib.import_module(f".{module}.experimental.{submodule}", package=__package__)
        classes = [m for _, m in inspect.getmembers(mod, inspect.isclass) if m.__module__ == mod.__name__ and pred(m)]
        assert len(classes) == 1, f"Multiple/no possible classes found in {module}.{submodule}:\n{classes}"
        cls = classes[0]
        return cls(**args_dict)

    latent = build_by_name(module="latent_spaces", submodule=args.latent_distribution)
    generator = build_by_name(
        module="models", submodule=args.generator, pred=lambda m: m.__name__.endswith("Generator")
    )
    discriminator = build_by_name(
        module="models", submodule=args.discriminator, pred=lambda m: m.__name__.endswith("Discriminator")
    )
    generator_losses = [
        build_by_name(module="losses", submodule=g_loss, pred=lambda m: m.__name__.startswith("Generator"))
        for g_loss in args.generator_losses
    ]
    discriminator_losses = [
        build_by_name(module="losses", submodule=d_loss, pred=lambda m: m.__name__.startswith("Discriminator"))
        for d_loss in args.discriminator_losses
    ]
    augmentations = [build_by_name(module="augmentation", submodule=aug) for aug in args.augmentations]

    # =============================================================
    # =========================== DATA ============================
    # =============================================================

    ffcv_preprocess = tvt.Compose(
        [tvt.Resize(args.preprocess_image_size, antialias=True), tvt.CenterCrop(args.preprocess_image_size)]
    )

    if args.random_crop:
        ffcv_decoder = RandomResizedCropRGBImageDecoder(
            (args.image_size, args.image_size),
            scale=(1 / args.random_crop_zoom, 1),
            ratio=(1 - args.random_crop_ratio, 1 + args.random_crop_ratio),
        )
    else:
        ffcv_decoder = SimpleRGBImageDecoder()

    class ToFloat(torch.nn.Module):
        def forward(self, x):
            return x.float().cuda()

    ffcv_pipeline = [
        ffcv_decoder,
        ToTensor(),
        ToTorchImage(),
        ToFloat(),
        tvt.Normalize([127.5] * 3, [127.5] * 3),
    ]
    if args.hflip:
        ffcv_pipeline.append(tvt.RandomHorizontalFlip())
    if args.vflip:
        ffcv_pipeline.append(tvt.RandomVerticalFlip())
    if args.random_rotate:
        # calculate minimum padding needed to ensure no non-padded pixels end up in image
        # no padding needed at 0, sqrt(2)*image_radius for 45 degrees or more
        padding = ceil(args.image_size * (1 - np.cos(4 * np.pi * min(args.random_rotate_degrees, 45) / 180)) / 4)
        ffcv_pipeline += [
            torch.nn.ReflectionPad2d((padding, padding, padding, padding)),
            tvt.RandomRotation(args.random_rotate_degrees, interpolation=tvt.InterpolationMode.BILINEAR, expand=True),
            tvt.CenterCrop(args.image_size),
        ]

    # =============================================================
    # ========================== TRAIN ============================
    # =============================================================

    LightningTrainer.from_argparse_args(
        args,
        callbacks=[
            TQDMProgressBar(),
            EarlyStopping(monitor=args.monitor_metric, patience=10 * args.test_kimg // args.epoch_kimg),
            ModelCheckpoint(
                monitor=args.monitor_metric,
                every_n_epochs=args.ckpt_kimg // args.epoch_kimg,
                save_top_k=args.ckpt_top_k,
                save_last=True,
            ),
            WeightsEMA(decay=args.ema_decay),
        ],
        max_epochs=args.kimg // args.epoch_kimg,
    ).fit(
        LightningGAN(
            latent=latent,
            generator=generator,
            discriminator=discriminator,
            discriminator_losses=discriminator_losses,
            generator_losses=generator_losses,
            shared_losses=[],
            augmentations=augmentations,
            ffcv_preprocess=ffcv_preprocess,
            ffcv_pipeline=ffcv_pipeline,
            lr_G=args.lr_G,
            lr_D=args.lr_D,
            n_D_steps=args.n_D_steps,
            cache_dir=args.cache_dir,
            input_dir=args.input_dir,
            batch_size=args.batch_size,
            jpeg_quality=args.jpeg_quality,
            num_workers=args.num_workers,
            epoch_kimg=args.epoch_kimg * 2,  # factor 2 because each optimizer_idx gets own batch
            test_kimg=args.test_kimg,
            monitor_metric=args.monitor_metric,
        )
    )
