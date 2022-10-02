import torch


class BaseDiffusionProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, prompts, t_start, t_end=1, verbose=True):
        pass
