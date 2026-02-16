import torch

from .efficient_quantile import _efficient_quantile


def quantile(tensor, q):
    return _efficient_quantile(tensor.cpu().flatten(), torch.FloatTensor([q]), True, 3).squeeze().to(tensor.device)


if __name__ == "__main__":
    example = torch.randn(17_000_000, device="cuda")
    try:
        torch.quantile(example, 0.99)
    except Exception as e:
        print(e)
    quantile(example, 0.99)
