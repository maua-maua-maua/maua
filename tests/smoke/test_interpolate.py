import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]


def test_rife_framerate(short_video, assert_plausible_image):
    from torchvision.io import read_video

    from maua.super.video.framerate.rife import interpolate, load_model

    frames = read_video(str(short_video), pts_unit="sec")[0][:2].permute(0, 3, 1, 2).float().div(255).cuda()
    model = load_model("RIFE-2.3", device="cuda")
    result = list(interpolate(frames[0:1], frames[1:2], model, factor=2))
    assert len(result) == 2, f"factor=2 should yield exactly 2 intermediate frames, got {len(result)}"
    f0, f1 = frames[0:1], frames[1:2]
    for mid in result:
        assert mid.shape == f0.shape
        assert_plausible_image(mid, lo=-0.1, hi=1.1)
        # an interpolant should lie between the endpoints, not off in space
        mae_mid = (mid - (f0 + f1) / 2).abs().mean()
        mae_ends = (f0 - f1).abs().mean()
        assert mae_mid <= mae_ends + 0.05, f"midframe deviates from endpoints: {mae_mid:.3f} vs {mae_ends:.3f}"
