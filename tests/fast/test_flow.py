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
