import contextlib
import os

import torch
import torch.nn as nn
from ....utility import download


@contextlib.contextmanager
def disable_gpu_fuser_on_pt19():
    # On PyTorch 1.9 a CUDA fuser bug prevents the Inception JIT model to run. See
    #   https://github.com/GaParmar/clean-fid/issues/5
    #   https://github.com/pytorch/pytorch/issues/64062
    # if torch.__version__.startswith("1.9."):
    old_val = torch._C._jit_can_fuse_on_gpu()
    torch._C._jit_override_can_fuse_on_gpu(False)
    yield
    # if torch.__version__.startswith("1.9."):
    torch._C._jit_override_can_fuse_on_gpu(old_val)


URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"


class Inception(nn.Module):
    def __init__(self, path="modelzoo/inception-2015-12-05.pt"):
        super().__init__()
        if not os.path.exists(path):
            download(URL, path)
        self.base = torch.jit.load(path).eval()
        self.layers = self.base.layers

    @torch.inference_mode()
    def forward(self, x):
        """
        Get the inception features without resizing
        x: Image with values in range [-1,1]
        """
        with disable_gpu_fuser_on_pt19():
            bs = x.shape[0]
            assert (x.shape[2] == 299) and (x.shape[3] == 299)
            features = self.layers.forward(x).view((bs, 2048))
            return features
