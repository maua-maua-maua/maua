import torch


class ModulationSum(torch.nn.Module):
    def __init__(self, modulated_modules):
        super().__init__()
        self.modulated_modules = modulated_modules

    def forward(self):
        # fmt:off
        return torch.sum([mod.forward() for mod in self.modulated_modules], dim=0) / \
               torch.sum([mod.modulation for mod in self.modulated_modules], dim=0)
        # fmt:on
