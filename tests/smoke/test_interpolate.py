import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]


def test_rife_framerate(short_video):
    from torchvision.io import read_video

    from maua.super.video.framerate.rife import interpolate, load_model

    frames = read_video(str(short_video), pts_unit="sec")[0][:2].permute(0, 3, 1, 2).float().div(255).cuda()
    model = load_model("RIFE-2.3", device="cuda")
    result = interpolate(frames[0:1], frames[1:2], model, factor=2)
    result = list(result) if not torch.is_tensor(result) else [result]
    assert len(result) >= 1
