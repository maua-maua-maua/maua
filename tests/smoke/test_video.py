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
    # video_sample returns a FramesOnDisk lazy dataset (what write_video consumes), not a tensor
    assert len(video) > 1
    frames = video[list(range(len(video)))]
    assert torch.is_tensor(frames) and frames.ndim == 4 and len(frames) == len(video)


@pytest.mark.xfail(
    reason="xformers memory-efficient attention has no fp32 kernel for this GPU's compute capability (sm_120); needs an fp16/bf16 pipeline",
    strict=False,
)
def test_temporalvideo_controlnet(short_video):
    from torchvision.io import read_video

    from maua.diffusion.temporalvideo_hf import stylize_video

    frames = read_video(str(short_video), pts_unit="sec")[0][:3].permute(0, 3, 1, 2).float().div(255)
    out = stylize_video(
        input_video=frames, prompt="an oil painting", num_steps=2, batch_size=2, height=64, width=64
    )
    assert torch.is_tensor(out) and out.shape[0] == frames.shape[0]


@pytest.mark.slow
@pytest.mark.backend_stable
def test_diffusion_interpolate(example_image, example_image2, tmp_path, tiny, assert_video):
    import shutil

    from maua.diffusion.interpolate import interpolate_images

    image_dir = tmp_path / "keyframes"
    image_dir.mkdir()
    shutil.copy(example_image, image_dir / "a.jpg")
    shutil.copy(example_image2, image_dir / "b.jpg")

    output = interpolate_images(
        image_dir=str(image_dir),
        prompt="a colorful painting",
        output_file=str(tmp_path / "interpolated.mp4"),
        n_frames=4,
        fps=tiny["fps"],
        size=tiny["size"],
        timesteps=tiny["steps"],
        batch_size=2,
    )
    assert_video(output, min_frames=3)


def test_cogvideo_import():
    # Full CogVideo generation needs a torch.distributed launch + three multi-GB models,
    # so it isn't a smoke test. But importing the module runs _import_cogvideo_globals(),
    # which is the whole rescue: it resolves the `models` package collision with GAN/pix2pix
    # and the icetk/protobuf shim. Assert those globals came back as real classes.
    import inspect

    from maua.autoregressive.cog.video import generate

    for name in ("CoglmStrategy", "CogVideoCacheModel", "DirectSuperResolution"):
        assert inspect.isclass(getattr(generate, name)), f"{name} did not resolve to a class"
