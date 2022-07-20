import numpy as np
import torch
from torch.autograd import grad

from . import Loss


class GeneratorPathLengthRegularization(Loss):
    def __init__(self, pl_weight, pl_interval, pl_decay, pl_batch_shrink, **kwargs) -> None:
        super().__init__()
        self.decay = pl_decay
        self.weight = pl_weight
        self.interval = pl_interval
        self.batch_shrink = pl_batch_shrink
        self.c = 0

        pl_mean = torch.ones(())
        self.register_buffer("pl_mean", pl_mean)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # fmt:off
        parser = parent_parser.add_argument_group("PathLengthRegularization")
        parser.add_argument("--pl_weight", type=float, default=2, help="Strength of path length regularization")
        parser.add_argument("--pl_batch_shrink", type=int, default=2, help="Factor to reduce batch size by for regularization calculation")
        parser.add_argument("--pl_decay", type=float, default=0.01, help="Exponential moving average decay of mean path length")
        parser.add_argument("--pl_interval", type=int, default=4, help="How often to apply regularization (lazy regularization)")
        # fmt:on
        return parent_parser

    def pre_G(self, latent, **kwargs):
        latent.requires_grad_()

    def forward(self, lightning_module, latent, fakes, preds_fake, **kwargs):
        self.c += 1
        if self.c % self.interval == 0:
            pl_noise = torch.randn_like(fakes) / np.sqrt(fakes.shape[2] * fakes.shape[3])
            pl_grads = grad(outputs=[(fakes * pl_noise).sum()], inputs=[latent], create_graph=True, only_inputs=True)[0]
            path_lengths = pl_grads.square()
            if path_lengths.dim() == 3:
                path_lengths = path_lengths.sum(2)
            path_lengths = path_lengths.mean(1).sqrt()

            pl_mean = torch.lerp(self.pl_mean, path_lengths.mean(), self.decay)
            self.pl_mean.copy_(pl_mean.detach())

            pl_penalty = (path_lengths - pl_mean).square()

            loss_G_pl = pl_penalty * self.weight

            lightning_module.log_dict(dict(pl_penalty=pl_penalty.mean(), loss_G_pl=loss_G_pl.mean()))

            return loss_G_pl
        else:
            return torch.zeros_like(preds_fake)
