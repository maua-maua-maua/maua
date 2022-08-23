import torch


class BaseDiffusionProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, prompts, start_step, n_steps, verbose):
        pass
