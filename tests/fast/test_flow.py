import numpy as np
import torch


def test_check_consistency_np():
    from maua.flow.consistency import check_consistency_np

    flow = np.zeros((16, 16, 2), dtype=np.float32)
    reliable = check_consistency_np(flow, flow)
    assert reliable.shape[:2] == (16, 16)


def test_check_consistency_torch():
    from maua.flow.consistency import check_consistency

    flow = torch.zeros(1, 16, 16, 2)
    reliable = check_consistency(flow, flow)
    assert torch.is_tensor(reliable)


def test_check_consistency_flags_inconsistent_flow():
    from maua.flow.consistency import check_consistency

    # Zero flow forward/backward are perfect inverses: everything reliable (~1).
    zero = torch.zeros(1, 32, 32, 2)
    reliable_zero = check_consistency(zero, zero)
    assert reliable_zero.float().mean() > 0.9, "consistent (zero) flow should be marked reliable"

    # A forward shift of +4px whose "backward" flow is also +4px (not -4px) does not round-trip,
    # so the interior must be flagged unreliable.
    fwd = torch.zeros(1, 32, 32, 2)
    fwd[..., 0] = 4.0
    bwd = fwd.clone()  # wrong sign on purpose
    reliable_bad = check_consistency(fwd, bwd)
    assert reliable_bad.float().mean() < reliable_zero.float().mean(), "inconsistent flow should be less reliable"
