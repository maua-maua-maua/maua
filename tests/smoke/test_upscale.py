import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]

# All of these are net 4x upscalers (the waifu2x/CARN 2x models are applied twice by module.upscale).
# One entry per distinct load_model/upscale code path: RealESRGAN, SwinIR (M+L), BSRGAN, RealSR,
# waifu2x (photo+anime), CARN, and latent-diffusion.
MODELS = [
    "RealESRGAN-x4plus",
    "RealESRGAN-x4plus-anime",
    "SwinIR-M-DFO-GAN",
    "SwinIR-L-DFOWMFC-GAN",
    "BSRGAN",
    "RealSR",
    "CARN",
    "waifu2x-photo-noise1",
    "waifu2x-anime-noise1",
    pytest.param("latent-diffusion", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("model_name", MODELS)
def test_upscale_image(small_image_64, model_name, assert_plausible_image):
    from maua.super.image.single import upscale_image

    out = upscale_image(str(small_image_64), model_name)
    assert torch.is_tensor(out)
    assert out.shape[-2:] == (256, 256), f"expected exact 4x upscale of 64px input, got {tuple(out.shape[-2:])}"
    # GAN upscalers (BSRGAN/SwinIR/RealSR) can overshoot [0, 1] by a couple tenths at edges.
    assert_plausible_image(out, lo=-0.3, hi=1.3)


def test_upscale_video_frame_by_frame(short_video, tmp_path):
    from decord import VideoReader

    from maua.super.video.frame_by_frame import upscale

    n_in = len(VideoReader(str(short_video)))
    result = upscale(str(short_video), "RealESRGAN-x4plus", "cuda", str(tmp_path))
    assert result is not None and len(result) == n_in, "output video should have one frame per input frame"
    h, w, _ = result[0].shape
    assert (h, w) == (512, 512), f"expected 4x upscale of the 128px clip, got {(h, w)}"
