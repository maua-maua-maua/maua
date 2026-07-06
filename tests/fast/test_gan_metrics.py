"""The precision/recall/density/coverage manifold metrics are pure torch — CPU-fast to verify."""

import torch


def test_prdc_identical_manifolds():
    from maua.GAN.metrics.prdc import prdc

    torch.manual_seed(0)
    feats = torch.randn(64, 16)
    # identical real/fake manifolds: precision and recall should be ~perfect
    precision, recall, density, coverage = prdc(feats, feats.clone(), nearest_k=5)
    for name, val in [("precision", precision), ("recall", recall), ("coverage", coverage)]:
        assert val > 0.9, f"{name}={float(val):.2f}, expected ~1 for identical manifolds"
    assert density > 0.5


def test_prdc_disjoint_manifolds():
    from maua.GAN.metrics.prdc import prdc

    torch.manual_seed(0)
    real = torch.randn(64, 16)
    fake = torch.randn(64, 16) + 100  # far-away cluster
    precision, recall, density, coverage = prdc(real, fake, nearest_k=5)
    assert precision < 0.1 and recall < 0.1, "disjoint manifolds should score near zero precision/recall"
