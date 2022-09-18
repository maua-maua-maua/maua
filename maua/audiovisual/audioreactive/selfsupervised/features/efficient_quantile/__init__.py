import torch

from .efficient_quantile import _efficient_quantile


def quantile(tensor, q):
    return _efficient_quantile(tensor.cpu().flatten(), torch.FloatTensor([q]), True, 3).squeeze().to(tensor.device)
