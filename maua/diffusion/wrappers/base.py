import torch


class DiffusionWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def set_target(self, target):
        pass

    def sample(self, image, start_step, stop_step, step_size):
        pass

    def forward(self):
        pass
