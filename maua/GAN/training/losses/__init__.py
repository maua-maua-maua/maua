import torch


class Loss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def pre_G(self, **kwargs):
        pass

    def pre_D(self, **kwargs):
        pass

    def forward(self, **Kwargs):
        return 0
