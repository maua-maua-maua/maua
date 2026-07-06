import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]

# The sniklaus neural optical-flow backends (each downloads its own weights on first use).
NEURAL_BACKENDS = ["pwc", "spynet", "liteflownet", "unflow"]


@pytest.mark.parametrize("backend", NEURAL_BACKENDS)
def test_neural_flow_backend(backend, short_video):
    from torchvision.io import read_video

    from maua.flow import get_flow_model

    frames = read_video(str(short_video), pts_unit="sec")[0][:2].permute(0, 3, 1, 2).float().div(255).cuda()
    model = get_flow_model([backend])
    flow = model(frames[0:1], frames[1:2])
    assert torch.is_tensor(flow)
    assert flow.shape[-1] == 2 or flow.shape[1] == 2, f"flow should have a 2-channel (u, v) axis, got {tuple(flow.shape)}"
    assert torch.isfinite(flow).all(), "optical flow contains NaN/Inf"


def test_farneback_flow_default(short_video):
    from torchvision.io import read_video

    from maua.flow import get_flow_model

    frames = read_video(str(short_video), pts_unit="sec")[0][:2].permute(0, 3, 1, 2).float().div(255).cuda()
    model = get_flow_model()  # default is the dependency-free farneback backend
    flow = model(frames[0:1], frames[1:2])
    assert torch.is_tensor(flow) and torch.isfinite(flow).all()
