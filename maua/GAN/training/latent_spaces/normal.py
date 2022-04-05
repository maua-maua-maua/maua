import torch


class NormalLatentDistribution(torch.nn.Module):
    def __init__(self, batch_size, nz, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.nz = nz
        self.dev = torch.nn.Parameter(torch.empty(0))

    def forward(self):
        return torch.randn((self.batch_size, self.nz), device=self.dev.device)
