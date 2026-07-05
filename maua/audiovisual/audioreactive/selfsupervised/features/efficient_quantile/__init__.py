import torch


def quantile(tensor, q):
    """Quantile that works on very large tensors.

    ``torch.quantile`` raises on inputs with more than ~16M elements, which is why this was
    originally a hand-rolled C++ extension. Absorbed here as a pure-torch fallback so no
    in-place build is needed: use ``torch.quantile`` when it is safe, otherwise a sort-based
    ``kthvalue`` (no size limit, stays on-device).
    """
    flat = tensor.flatten()
    q = float(q)
    if flat.numel() <= 16_000_000:
        return torch.quantile(flat, q)
    k = int(round(q * (flat.numel() - 1))) + 1
    k = max(1, min(flat.numel(), k))
    return flat.kthvalue(k).values


if __name__ == "__main__":
    example = torch.randn(17_000_000, device="cuda" if torch.cuda.is_available() else "cpu")
    print(quantile(example, 0.99))
