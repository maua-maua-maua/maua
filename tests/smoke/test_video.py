import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]


@pytest.mark.backend_stable
def test_video_diffusion(short_video, tiny):
    from maua.diffusion.video import video_sample

    video = video_sample(
        diffusion="stable",
        init=str(short_video),
        text="a watercolor painting",
        size=(tiny["size"], tiny["size"]),
        timesteps=tiny["steps"],
        first_skip=0.4,
        skip=0.7,
    )
    assert torch.is_tensor(video) and video.ndim == 4 and len(video) > 1


def test_temporalvideo_controlnet(short_video):
    from torchvision.io import read_video

    from maua.diffusion.temporalvideo_hf import stylize_video

    frames = read_video(str(short_video), pts_unit="sec")[0][:3].permute(0, 3, 1, 2).float().div(255)
    out = stylize_video(
        input_video=frames, prompt="an oil painting", num_steps=2, batch_size=2, height=64, width=64
    )
    assert torch.is_tensor(out) and out.shape[0] == frames.shape[0]


@pytest.mark.skip(reason="maua/diffusion/interpolate.py is script-style with no callable API; wrap in Phase B/C")
def test_diffusion_interpolate():
    pass


@pytest.mark.slow
@pytest.mark.skip(reason="CogVideo weights are enormous; triage in Phase B")
def test_cogvideo():
    pass
