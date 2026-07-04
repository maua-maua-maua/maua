import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.download]


def test_img2img(example_image, tiny):
    from maua.diffusion.image import image_sample

    img = image_sample(
        init=str(example_image),
        text="a watercolor painting",
        sizes=[(tiny["size"], tiny["size"])],
        skips=[0.5],
        timesteps=4,
        diffusion="stable",
        sampler="plms",
    )
    assert torch.is_tensor(img) and img.ndim == 4


@pytest.mark.backend_stable
def test_outpaint(example_image):
    from maua.diffusion.outpaint import OutpaintingStableDiffusion
    from maua.prompt import ImagePrompt, TextPrompt

    diffusion = OutpaintingStableDiffusion(sampler="euler_ancestral")
    img = ImagePrompt(path=str(example_image), size=(64, 64)).img.cuda()
    out = diffusion.outpaint(img, [TextPrompt("a wide landscape")], t_start=0.4)
    assert torch.is_tensor(out)
    assert out.shape[-1] * out.shape[-2] > img.shape[-1] * img.shape[-2]
