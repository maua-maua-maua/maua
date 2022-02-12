import torch
import traceback


class ModulationSum(torch.nn.Module):
    def __init__(self, modulated_modules):
        super().__init__()
        self.modulated_modules = modulated_modules

    def forward(self):
        average, weight = None, torch.zeros([1])
        for mod in self.modulated_modules:
            try:
                weight += mod.modulation[mod.index % len(mod.modulation)]
                if average is None:
                    average = mod.forward().squeeze()
                else:
                    average += mod.forward().squeeze()
            except Exception as e:
                print()
                print(f"ERROR '{e}' from {mod.__class__.__name__}")
                traceback.print_exc()
                exit()
        try:
            return (average / weight).float().unsqueeze(0)
        except:
            return None
