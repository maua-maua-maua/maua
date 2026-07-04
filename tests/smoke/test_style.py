import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]


def test_style_transfer_image(example_image, example_image2):
    from maua.style.image import transfer

    result = transfer(
        content_img=str(example_image),
        style_imgs=[str(example_image2)],
        size=128,
        n_iters=4,
    )
    assert torch.is_tensor(result) and result.ndim == 4


@pytest.mark.slow
def test_style_transfer_video(short_video, example_image2):
    from maua.style.video import transfer

    result = transfer(
        content_video=str(short_video),
        style_imgs=[str(example_image2)],
        size=128,
        n_iters=4,
        n_passes=1,
        flow_models=["farneback"],
    )
    assert result is not None
