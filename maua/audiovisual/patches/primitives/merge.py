import torch


class ModulationSum(torch.nn.Module):
    def __init__(self, modulated_modules):
        super().__init__()
        self.modulated_modules = modulated_modules

    def forward(self):
        average, weight = None, 0
        for mod in self.modulated_modules:
            try:
                weight += mod.modulation[mod.index % len(mod.modulation)]
                if average is None:
                    average = mod.forward().squeeze()
                else:
                    average += mod.forward().squeeze()
            except Exception as e:
                print(e)
                print()
                print(mod.__class__.__name__)
                exit()
        return (average / weight).unsqueeze(0)
