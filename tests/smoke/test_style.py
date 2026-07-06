import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]


def test_style_transfer_image(example_image, example_image2, assert_plausible_image):
    from maua.style.image import transfer

    result = transfer(
        content_img=str(example_image),
        style_imgs=[str(example_image2)],
        size=128,
        n_iters=4,
    )
    assert torch.is_tensor(result) and result.ndim == 4
    assert_plausible_image(result, lo=-0.1, hi=1.1)


def test_style_image_multires(example_image, example_image2, assert_plausible_image, device):
    """Multi-resolution style transfer: the coarse pass output seeds the fine pass."""
    from maua.ops.io import img2tensor

    from PIL import Image

    from maua.style.image_multires import transfer_multires

    result = transfer_multires(
        content_img=img2tensor(Image.open(example_image)),
        style_imgs=[img2tensor(Image.open(example_image2))],
        init_img=None,
        init_type="content",
        match_hist="avg",
        sizes=[96, 128],
        parameterization="rgb",
        perceptor="kbc-vgg19",
        perceptor_kwargs={},
        optimizer="LBFGS",
        lr=0.05,
        optimizer_kwargs={},
        n_iters=[3, 3],
        content_weight=1,
        style_weight=50,
        style_scale=1,
        device=device,
    )
    assert torch.is_tensor(result) and result.ndim == 4
    assert_plausible_image(result, lo=-0.1, hi=1.1)


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


def test_omnimae_constructs_and_runs():
    """The OmniMAE ViT builders exercise a lot of rescued factory/hydra code. Build the base
    pretraining model (random init, no checkpoint) and push a tiny clip through the trunk."""
    from maua.style.omnimae import vit_base_mae_pretraining

    model = vit_base_mae_pretraining().cuda().eval()
    # OmniMAE ingests video as [B, C, T, H, W]; the patch embed pads T then tokenizes 16x16 patches.
    clip = torch.randn(1, 3, 16, 224, 224, device="cuda")
    with torch.no_grad():
        out = model.trunk(clip, mask=None)
    out = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.is_tensor(out) and torch.isfinite(out).all()
