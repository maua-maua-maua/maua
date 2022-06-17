import torch
from torch.autograd import grad

from . import Loss


class DiscriminatorR1Penalty(Loss):
    def __init__(self, batch_size, image_size, r1_gamma, r1_interval, **kwargs) -> None:
        super().__init__()
        self.interval = r1_interval
        self.gamma = r1_gamma
        if self.gamma is None:
            self.gamma = 2e-4 * image_size**2 / batch_size
        self.gamma /= 2  # why does stylegan2 use this?
        self.c = 0

    @staticmethod
    def add_model_specific_args(parent_parser):
        # fmt:off
        parser = parent_parser.add_argument_group("R1Penalty")
        parser.add_argument("--r1_gamma", type=float, default=None, help="Strength of R1 penalty, None uses heuristic introduced in StyleGAN2-ADA")
        parser.add_argument("--r1_interval", type=int, default=16, help="How often to apply penalty (lazy regularization)")
        # fmt:on
        return parent_parser

    def pre_D(self, reals, **kwargs):
        reals.requires_grad_()

    def forward(self, lightning_module, preds_real, reals, **kwargs):
        self.c += 1
        if self.c % self.interval == 0:
            r1_grads = grad(outputs=[preds_real.sum()], inputs=[reals], create_graph=True, only_inputs=True)[0]
            penalty = r1_grads.square().sum((1, 2, 3))
            loss_D_r1 = penalty * self.gamma

            lightning_module.log_dict(dict(r1_penalty=penalty.mean(), loss_D_r1=loss_D_r1.mean()))

            return loss_D_r1
        else:
            return torch.zeros_like(preds_real)
