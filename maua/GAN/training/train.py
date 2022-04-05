from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from copy import deepcopy
from glob import glob
from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
import torch.multiprocessing as mp
import torchvision as tv
import torchvision.transforms as tvt
from average import EWMA
from ffcv.fields import RGBImageField
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToDevice, ToTensor, ToTorchImage
from ffcv.writer import DatasetWriter
from overrides import overrides
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.callbacks import Callback as LightningCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import Module as TorchModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.functional import resize

from ..metrics.compute import compute as compute_metrics


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

    try:
        loader = construct_loader()
        rebuild = False
    except:
        rebuild = True

    if rebuild:

        class FFCVPreprocessorDataset(TorchDataset):
            def __len__(self):
                return len(files)

            def __getitem__(self, idx):
                return np.asarray(ffcv_preprocess(PIL.Image.open(files[idx]).convert("RGB")))[np.newaxis]

        data = FFCVPreprocessorDataset()
        DatasetWriter(
            cache_path, {"image": RGBImageField(max_resolution=max(data[0].shape), jpeg_quality=jpeg_quality)}
        ).from_indexed_dataset(data)

        loader = construct_loader()

    iterator = Iterator(loader, epoch_kimg)
    iterator.path = cache_path

    return iterator


class WeightsEMA(LightningCallback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn"t the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn"t need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """

    def __init__(self, decay: float = 0.9999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don"t want to return their parameters.

        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()

    @overrides
    def on_train_start(self, trainer: LightningTrainer, pl_module: LightningModule) -> None:
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.ema_device:
                self.ema_state_dict = {
                    k: tensor.to(device=self.ema_device) for k, tensor in self.ema_state_dict.items()
                }

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

        self._ema_state_dict_ready = True

    @rank_zero_only
    def on_train_batch_end(self, trainer: LightningTrainer, pl_module: LightningModule, *args, **kwargs) -> None:
        # Update EMA weights
        with torch.no_grad():
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(self.decay * ema_value + (1.0 - self.decay) * value, non_blocking=True)

    @overrides
    def on_validation_start(self, trainer: LightningTrainer, pl_module: LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
        pl_module.trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), (
            f"There are some keys missing in the ema static dictionary broadcasted. "
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        )
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    @overrides
    def on_validation_end(self, trainer: LightningTrainer, pl_module: LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        # Replace EMA weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=False)

    @overrides
    def on_save_checkpoint(
        self, trainer: LightningTrainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> dict:
        return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

    @overrides
    def on_load_checkpoint(
        self, trainer: LightningTrainer, pl_module: LightningModule, callback_state: Dict[str, Any]
    ) -> None:
        self._ema_state_dict_ready = callback_state["_ema_state_dict_ready"]
        self.ema_state_dict = callback_state["ema_state_dict"]


class LightningGAN(LightningModule):
    def __init__(
        self,
        # modules
        latent: TorchModule,
        generator: TorchModule,
        discriminator: TorchModule,
        discriminator_losses: List[TorchModule],
        generator_losses: List[TorchModule],
        shared_losses: List[TorchModule],
        # settings
        batch_size: int,
        lr_G: float,
        lr_D: float,
        n_D_steps: int,
        # data
        input_dir: str,
        ffcv_preprocess: Callable,
        ffcv_pipeline: Callable,
        cache_dir: str,
        num_workers: int,
        jpeg_quality: int,
        epoch_kimg: int,
        test_kimg: int,
        **kwargs,
    ):
        super().__init__()

        self.latent = latent
        self.G = generator
        self.D = discriminator

        self.losses_D = discriminator_losses
        self.losses_G = generator_losses
        self.losses_shared = shared_losses

        self.batch_size = batch_size
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.n_D_steps = n_D_steps

        self.ffcv_preprocess = ffcv_preprocess
        self.ffcv_pipeline = ffcv_pipeline
        self.data_cache_path = f"{cache_dir}/{Path(input_dir).stem}_ffcv.beton"
        self.files = sum([glob(f"{input_dir}/*{ext}") for ext in tv.datasets.folder.IMG_EXTENSIONS], [])
        self.num_workers = num_workers
        self.jpeg_quality = jpeg_quality

        self.epoch_kimg = epoch_kimg
        self.test_kimg = test_kimg

        self.metric_emas = {}

        self.save_hyperparameters()
        self.train_dataloader()

    def forward(self):
        return self.G(self.latent()).add(1).div(2).clamp(0, 1)

    def train_dataloader(self):
        return ImageLoader(
            self.files,
            self.ffcv_preprocess,
            self.ffcv_pipeline,
            self.data_cache_path,
            self.epoch_kimg,
            self.batch_size,
            self.num_workers,
            self.jpeg_quality,
        )

    def val_dataloader(self):
        return torch.ones(1)

    def training_step(self, reals, _, optimizer_idx):

        if optimizer_idx == 0:  # G step
            latent = self.latent()
            fakes = self.G(latent)
            fake_preds = self.D(fakes)
            kwargs = dict(lightning_module=self, latent=latent, fakes=fakes, fake_preds=fake_preds)
            losses = [loss(**kwargs) for loss in self.losses_G]
            return sum(losses)

        if optimizer_idx == 1:  # D step
            latent = self.latent()
            fakes = self.G(latent)
            fake_preds = self.D(fakes)
            real_preds = self.D(reals)
            kwargs = dict(
                lightning_module=self,
                latent=latent,
                fakes=fakes,
                reals=reals,
                fake_preds=fake_preds,
                real_preds=real_preds,
            )
            losses = [loss(**kwargs) for loss in self.losses_D]
            return sum(losses)

    def validation_step(self, batch, batch_idx):
        imgs = torch.cat([self.forward() for _ in range(ceil(16 * 9 / self.batch_size))])[: 16 * 9]
        grid = tv.utils.make_grid(imgs, nrow=16, padding=0)
        if grid.shape[-1] > 4096:
            grid = resize(grid, (4096 // 16 * 9, 4096), antialias=True)
        self.logger.experiment.add_image("Example Images", grid, self.global_step)

        if (self.current_epoch * self.epoch_kimg) % self.test_kimg == 0:
            metric_val = compute_metrics(
                self.train_dataloader(),
                self.forward,
                n_samples=len(self.files),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                verbose=True,
            )

            # get EMA versions of each metric
            for key, val in metric_val.items():
                key_ema = f"{key} EMA"
                if not key_ema in self.metric_emas:
                    self.metric_emas[key_ema] = EWMA(beta=0.9)
                self.metric_emas[key_ema].update(val)
            metric_emas = {key: avg.get() for key, avg in self.metric_emas.items()}

            self.log_dict(metric_val)
            fsd = metric_emas["Frechet SwAV Distance"]
            del metric_emas["Frechet SwAV Distance"]
            self.log_dict(metric_emas)
            self.log_dict({"Frechet SwAV Distance": fsd}, prog_bar=True)

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr_G, betas=(0.5, 0.99))
        scheduler_G = ReduceLROnPlateau(optimizer_G)
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr_D, betas=(0.5, 0.99))
        scheduler_D = ReduceLROnPlateau(optimizer_D)
        return (
            {
                "optimizer": optimizer_G,
                "lr_scheduler": {"scheduler": scheduler_G, "monitor": "Frechet SwAV Distance"},
                "frequency": 1,
            },
            {
                "optimizer": optimizer_D,
                "lr_scheduler": {"scheduler": scheduler_D, "monitor": "Frechet SwAV Distance"},
                "frequency": self.n_D_steps,
            },
        )


def main(args):
    dict_args = vars(args)

    # =============================================================
    # ========================= GENERATOR =========================
    # =============================================================

    if args.generator == "deepconvolutional":
        generator = DeepConvolutionalGenerator(**dict_args)

    # =============================================================
    # ==================== LATENT DISTRIBUTION ====================
    # =============================================================

    if args.latent_distribution == "normal":
        latent = NormalLatentDistribution(**dict_args)

    # =============================================================
    # ===================== GENERATOR LOSSES ======================
    # =============================================================

    generator_losses = []

    if "cross_entropy" in args.generator_losses:
        generator_losses.append(GeneratorCrossEntropy(logits=args.logits))

    # =============================================================
    # ======================= DISCRIMINATOR =======================
    # =============================================================

    if args.discriminator == "deepconvolutional":
        discriminator = DeepConvolutionalDiscriminator(**dict_args)

    # =============================================================
    # =================== DISCRIMINATOR LOSSES ====================
    # =============================================================

    discriminator_losses = []

    if "cross_entropy" in args.discriminator_losses:
        discriminator_losses.append(DiscriminatorCrossEntropy(logits=args.logits))

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
            tvt.RandomRotation(args.random_rotate_degrees, interpolation=tvt.InterpolationMode.BILINEAR),
            tvt.Resize(args.image_size),
        ]

    # =============================================================
    # ========================== TRAIN ============================
    # =============================================================

    LightningTrainer.from_argparse_args(
        args,
        callbacks=[
            TQDMProgressBar(),
            EarlyStopping(monitor="Frechet SwAV Distance", patience=10 * args.test_kimg // args.epoch_kimg),
            ModelCheckpoint(
                monitor="Frechet SwAV Distance",
                every_n_epochs=args.test_kimg // args.epoch_kimg,
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
            epoch_kimg=args.epoch_kimg,
            test_kimg=args.test_kimg,
        )
    )


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)

    # fmt: off
    subparser = parser.add_argument_group("Models", description="Settings related to the neural network models")
    subparser.add_argument("-L", "--latent_distribution", type=str, default="normal", help="which input latent distribution to use")
    subparser.add_argument("-G", "--generator", type=str, default="deepconvolutional", help="which generator to use")
    subparser.add_argument("-D", "--discriminator", type=str, default="deepconvolutional", help="which discriminator to use")
    subparser.add_argument("-EMA", "--ema_decay", type=float, default=0.995, help="model weight exponential moving average decay")

    subparser = parser.add_argument_group("Losses", description="Settings related to losses used to train the models")
    subparser.add_argument("-DL", "--discriminator_losses", type=str, nargs="+", default=["cross_entropy"], help="which discriminator losses to use")
    subparser.add_argument("-GL", "--generator_losses", type=str, nargs="+", default=["cross_entropy"], help="which generator losses to use")
    
    subparser = parser.add_argument_group("Optimization", description="Settings related optimizers")
    subparser.add_argument("--lr_G", type=float, default=2e-4, help="Starting learning rate for the generator")
    subparser.add_argument("--lr_D", type=float, default=2e-4, help="Starting learning rate for the discriminator")
    subparser.add_argument("--n_D_steps", type=int, default=1, help="How many discriminator steps to perform per generator step")

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

    subparser = parser.add_argument_group("Training", description="General training and logging settings")
    subparser.add_argument("--kimg", type=int, default=100_000, help="How many thousands of images to train on in total")
    subparser.add_argument("--epoch_kimg", type=int, default=10, help="How many kimg per epoch")
    subparser.add_argument("--ckpt_kimg", type=int, default=10, help="How many thousands of images to train on between model checkpoints")
    subparser.add_argument("--ckpt_top_k", type=int, default=10, help="How many checkpoints to keep")
    subparser.add_argument("--test_kimg", type=int, default=1000, help="How many epochs to wait between running metric analysis")

    subparser = parser.add_argument_group("Training augmentations", description="Settings related to augmentations for improving training (augmentations applied here will NOT be visible in output data)")

    subparser = parser.add_argument_group("Dataloading", description="Settings related to caching/loading data during training")
    subparser.add_argument("--batch_size", type=int, default=32, help="Directory containing image files to train on")
    subparser.add_argument("--num_workers", type=int, default=8, help="Number of background thread workers loading images")
    subparser.add_argument("--jpeg_quality", type=int, default=95, help="Quality of preprocessed jpegs stored in cache. Lower values will save disk space at the cost of image quality.")
    subparser.add_argument("--cache_dir", type=str, default="cache/", help="Directory to cache preprocessed data in")
    # fmt: on

    temp_args, _ = parser.parse_known_args()
    override_args = {}

    # =============================================================
    # ========================= GENERATOR =========================
    # =============================================================

    if temp_args.generator == "deepconvolutional":
        from .models.deepconvolutional import DeepConvolutionalGenerator

        parser = DeepConvolutionalGenerator.add_model_specific_args(parser)
    elif temp_args.generator == "stylehypermixerfly":
        from .models.stylehypermixerfly import StyleHyperMixerFlyGenerator

        parser = StyleHyperMixerFlyGenerator.add_model_specific_args(parser)
    else:
        raise ValueError(f"Unknown generator {temp_args.generator}")

    # =============================================================
    # ==================== LATENT DISTRIBUTION ====================
    # =============================================================

    if temp_args.latent_distribution == "normal":
        from .latent_spaces.normal import NormalLatentDistribution

    # =============================================================
    # ===================== GENERATOR LOSSES ======================
    # =============================================================

    if "cross_entropy" in temp_args.discriminator_losses:
        from .losses.cross_entropy import GeneratorCrossEntropy

    # =============================================================
    # ======================= DISCRIMINATOR =======================
    # =============================================================

    if temp_args.discriminator == "deepconvolutional":
        from .models.deepconvolutional import DeepConvolutionalDiscriminator

        parser = DeepConvolutionalDiscriminator.add_model_specific_args(parser)
        override_args["logits"] = False
    elif temp_args.discriminator == "stylehypermixerfly":
        from .models.stylehypermixerfly import HyperMixerFlyDiscriminator

        parser = HyperMixerFlyDiscriminator.add_model_specific_args(parser)
        override_args["logits"] = True
    else:
        raise ValueError(f"Unknown discriminator {temp_args.discriminator}")

    # =============================================================
    # =================== DISCRIMINATOR LOSSES ====================
    # =============================================================

    if "cross_entropy" in temp_args.discriminator_losses:
        from .losses.cross_entropy import DiscriminatorCrossEntropy

    parser = LightningTrainer.add_argparse_args(parser)
    parser.add_argument("-h", "--help", action="help")
    args = parser.parse_args()

    for k, v in override_args.items():
        setattr(args, k, v)

    main(args)
