"""
Shims for modules that old third-party dependencies (submodules, basicsr, ic_gan, ...) import
from locations that no longer exist in current torchvision. Installed by maua/__init__.py so any
entrypoint into the package gets them.
"""

import sys
import types


def install_shims():
    import torch.hub
    import torchvision.models
    import torchvision.transforms.functional

    # torchvision.models.utils.load_state_dict_from_url moved to torch.hub (torchvision 0.13)
    if not hasattr(torchvision.models, "utils"):
        mod = types.ModuleType("torchvision.models.utils")
        mod.load_state_dict_from_url = torch.hub.load_state_dict_from_url
        torchvision.models.utils = mod
        sys.modules["torchvision.models.utils"] = mod

    # torchvision.transforms.functional_tensor was removed in torchvision 0.17 (basicsr needs it)
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        try:
            import torchvision.transforms.functional_tensor  # noqa: F401
        except ImportError:
            mod = types.ModuleType("torchvision.transforms.functional_tensor")
            mod.rgb_to_grayscale = torchvision.transforms.functional.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = mod
