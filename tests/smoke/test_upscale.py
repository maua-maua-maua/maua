import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]

MODELS = [
    "RealESRGAN-x4plus",
    "SwinIR-M-DFO-GAN",
    "BSRGAN",
    "waifu2x-photo-noise1",
    pytest.param("latent-diffusion", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("model_name", MODELS)
def test_upscale_image(small_image_64, model_name):
    from maua.super.image.single import upscale_image

    out = upscale_image(str(small_image_64), model_name)
    assert torch.is_tensor(out)
    assert min(out.shape[-2:]) > 64


def test_upscale_video_frame_by_frame(short_video, tmp_path):
    from maua.super.video.frame_by_frame import upscale

    result = upscale(str(short_video), "RealESRGAN-x4plus", "cuda", str(tmp_path))
    if result is not None:  # generator-style: consume a couple of frames
        for i, _frame in enumerate(result):
            if i >= 1:
                break
    assert any(tmp_path.iterdir()), "expected upscaled output in out_dir"
