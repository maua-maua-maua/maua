from math import floor

import torch
from kornia.filters import gaussian_blur2d


class InitialBlur(torch.nn.Module):
    def __init__(self, batch_size, blur_init_sigma, blur_fade_kimg, **kwargs) -> None:
        super().__init__()
        self.init_sigma = blur_init_sigma
        self.fade_kimg = batch_size / 32 * blur_fade_kimg
        self.batch_size = batch_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        # fmt:off
        parser = parent_parser.add_argument_group("InitialBlur")
        parser.add_argument("--blur_init_sigma", type=float, default=10, help="Strength of initial blur at start of training")
        parser.add_argument("--blur_fade_kimg", type=int, default=200, help="How many thousands of images to fade out blurring at start of training")
        # fmt:on
        return parent_parser

    def forward(self, lightning_module, reals, fakes, **kwargs):
        blur_sigma = (
            max(1 - (lightning_module.global_step * self.batch_size) / (self.fade_kimg * 1e3), 0) * self.init_sigma
            if self.fade_kimg > 0
            else 0
        )
        if blur_sigma > 0:
            blur_size = floor(blur_sigma * 3)
            blur_size = blur_size + (1 - blur_size % 2)
            if reals is not None:
                reals = gaussian_blur2d(reals, kernel_size=(blur_size, blur_size), sigma=(blur_sigma, blur_sigma))
            fakes = gaussian_blur2d(fakes, kernel_size=(blur_size, blur_size), sigma=(blur_sigma, blur_sigma))
        return reals, fakes
