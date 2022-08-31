import argparse
import datetime
import glob
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import center_crop, resize, to_tensor
from transformers import logging

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/../submodules/stable_diffusion")
from ..submodules.stable_diffusion.ldm.util import instantiate_from_config

logging.set_verbosity_error()


class ImageLogger(Callback):
    def __init__(self, batch_frequency, num_examples):
        super().__init__()
        self.batch_freq = batch_frequency
        self.num_examples = num_examples
        self.log_images_kwargs = {
            "use_ema_scope": False,
            "inpaint": False,
            "plot_progressive_rows": False,
            "plot_diffusion_rows": False,
            "N": self.num_examples,
            "unconditional_guidance_scale": 3.0,
            "unconditional_guidance_label": [""],
        }

    @rank_zero_only
    def log_local(self, save_dir, images, global_step, current_epoch, batch_idx):
        grid = torchvision.utils.make_grid(images, nrow=4)
        grid = grid.clamp(-1.0, 1.0).add(1.0).div(2.0)  # -1,1 -> 0,1; c,h,w
        grid = grid.permute(1, 2, 0)
        grid = grid.cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        filename = "samples_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.inference_mode():
            images = torch.cat(
                [
                    pl_module.log_images(batch, split="train", **self.log_images_kwargs)["samples"]
                    for _ in range(self.num_examples // batch["image"].shape[0])
                ]
            )
            self.log_local(pl_module.logdir, images, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (
            pl_module.global_step % self.batch_freq == 0
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.num_examples > 0
        ):
            self.log_img(pl_module, batch, batch_idx)


def worker_init_fn(_):
    return np.random.seed(np.random.get_state()[1][0] + torch.utils.data.get_worker_info().id)


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys."""

    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])

    return result


class Text2ImageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, path) -> None:
        super().__init__()
        self.dataset = Text2ImageDataset(path)
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=batch_size,
            shuffle=True,
            collate_fn=dict_collation_fn,
            worker_init_fn=worker_init_fn,
        )
        self.datasets = {
            "train": self.dataset,
            "validation": self.dataset,
            "test": self.dataset,
            "predict": self.dataset,
        }

    def __iter__(self):
        return iter(self.loader)

    def train_dataloader(self):
        return self.loader


class Text2ImageDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.images = sum([glob(f"{path}/*{ext}") for ext in torchvision.datasets.folder.IMG_EXTENSIONS], [])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = to_tensor(resize(center_crop(img, min(img.size)), 512, antialias=True)).permute(1, 2, 0).mul(2).sub(1)
        return {"caption": "", "image": img}


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-d", "--datadir", type=str, help="path to directory of images")
    parser.add_argument("-n", "--name", type=str, help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, help="path to .ckpt to resume")
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-5, help="training learning rate")
    parser.add_argument("-s", "--seed", type=int, default=42, help="seed for seed_everything")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging")
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="training batch size")
    parser.add_argument("-acc", "--accumulate-batches", type=int, default=1, help="batches to accumulate per step")
    parser.add_argument("-ne", "--num-examples", type=int, default=8, help="number of images to sample per log step")
    parser.add_argument("-le", "--log-every", type=int, default=2500, help="number of steps per image sample log")
    parser.add_argument("-se", "--save-every", type=int, default=5000, help="number of steps per saved checkpoint")
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    name = opt.name if opt.name else Path(opt.datadir).stem
    logdir = os.path.join(opt.logdir, f"stable_{now}_{name}")
    seed_everything(opt.seed)

    # ==================================================================================================================
    # =============================================== Trainer Setup ====================================================
    # ==================================================================================================================

    config = OmegaConf.load(
        os.path.abspath(os.path.dirname(__file__))
        + "/../submodules/stable_diffusion/configs/stable-diffusion/v1-inference.yaml"
    )

    trainer_config = OmegaConf.create(
        {"accelerator": "ddp", "benchmark": True, "limit_val_batches": 0, "num_sanity_val_steps": 0}
    )
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    trainer_config.accumulate_grad_batches = opt.accumulate_batches
    trainer_opt = argparse.Namespace(**trainer_config)

    trainer_kwargs = dict()
    trainer_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
    callbacks_cfg = {
        "image_logger": {
            "target": "maua.diffusion.finetune_stable.ImageLogger",
            "params": {"batch_frequency": opt.log_every, "num_examples": opt.num_examples},
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {"logging_interval": "step"},
        },
        "checkpoint_callback": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": logdir,
                "filename": "{epoch:06}-{step:09}",
                "verbose": True,
                "save_top_k": -1,
                "every_n_train_steps": opt.save_every,
                "save_weights_only": True,
            },
        },
    }
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    # ==================================================================================================================
    # ================================================ Model Setup =====================================================
    # ==================================================================================================================

    config.model.params.first_stage_key = "image"
    config.model.params.cond_stage_key = "caption"
    config.model.base_learning_rate = opt.learning_rate

    if opt.resume:
        model = load_model_from_config(config, opt.resume)
    else:
        model = instantiate_from_config(config.model)

    n_gpu = len(trainer_config.gpus.strip(",").split(","))
    model.learning_rate = opt.accumulate_batches * n_gpu * opt.batch_size * opt.learning_rate
    model.logdir = logdir

    # ==================================================================================================================
    # ================================================== Train! ========================================================
    # ==================================================================================================================

    os.makedirs(logdir, exist_ok=True)
    try:
        trainer.fit(model, Text2ImageDataModule(opt.batch_size, opt.datadir))
    except:
        trainer.save_checkpoint(os.path.join(logdir, "last.ckpt"), weights_only=True)
        raise
