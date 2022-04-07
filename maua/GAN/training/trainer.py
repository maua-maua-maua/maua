from copy import deepcopy
from glob import glob
from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torchvision as tv
from average import EWMA
from overrides import overrides
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.callbacks import Callback as LightningCallback
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import Module as TorchModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms.functional import resize

from ..metrics.compute import compute as compute_metrics
from .dataset.image import ImageLoader


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
        augmentations: List[TorchModule],
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
        monitor_metric: str,
        **kwargs,
    ):
        super().__init__()

        self.latent = latent
        self.G = generator
        self.D = discriminator

        self.losses_D = torch.nn.ModuleList(discriminator_losses)
        self.losses_G = torch.nn.ModuleList(generator_losses)
        self.losses_shared = torch.nn.ModuleList(shared_losses)
        self.augmentations = torch.nn.ModuleList(augmentations)

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
        self.monitor_metric = monitor_metric

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

            for loss in self.losses_G:
                loss.pre_G(latent=latent)

            fakes = self.G(latent)

            for loss in self.losses_G:
                loss.pre_D(latent=latent, fakes=fakes)

            for aug in self.augmentations:
                reals, fakes = aug(lightning_module=self, reals=reals, fakes=fakes)

            preds_fake = self.D(fakes)

            losses = [
                loss(lightning_module=self, latent=latent, fakes=fakes, preds_fake=preds_fake) for loss in self.losses_G
            ]
            return sum(losses).mean()

        if optimizer_idx == 1:  # D step
            latent = self.latent()

            for loss in self.losses_D:
                loss.pre_G(latent=latent, reals=reals)

            fakes = self.G(latent)

            for loss in self.losses_D:
                loss.pre_D(latent=latent, fakes=fakes, reals=reals)

            for aug in self.augmentations:
                reals, fakes = aug(lightning_module=self, reals=reals, fakes=fakes)

            preds_fake = self.D(fakes)
            preds_real = self.D(reals)

            losses = [
                loss(
                    lightning_module=self,
                    latent=latent,
                    fakes=fakes,
                    reals=reals,
                    preds_fake=preds_fake,
                    preds_real=preds_real,
                )
                for loss in self.losses_D
            ]
            return sum(losses).mean()

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

            val = metric_val[self.monitor_metric]
            del metric_val[self.monitor_metric]
            self.log_dict({self.monitor_metric: val}, prog_bar=True)
            self.log_dict(metric_val)
            self.log_dict(metric_emas)

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr_G, betas=(0.5, 0.99))
        scheduler_G = ReduceLROnPlateau(optimizer_G)
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr_D, betas=(0.5, 0.99))
        scheduler_D = ReduceLROnPlateau(optimizer_D)
        return (
            {
                "optimizer": optimizer_G,
                "lr_scheduler": {"scheduler": scheduler_G, "monitor": self.monitor_metric},
                "frequency": 1,
            },
            {
                "optimizer": optimizer_D,
                "lr_scheduler": {"scheduler": scheduler_D, "monitor": self.monitor_metric},
                "frequency": self.n_D_steps,
            },
        )
