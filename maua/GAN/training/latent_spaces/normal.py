import torch


class NormalLatentDistribution(torch.nn.Module):
    def __init__(self, batch_size, z_dim, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.dev = torch.nn.Parameter(torch.empty(0))

    def forward(self):
        return torch.randn((self.batch_size, self.z_dim), device=self.dev.device)
