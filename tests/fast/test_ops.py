import torch

from maua.ops.image import resample
from maua.ops.io import img2tensor, load_image, tensor2img
from maua.ops.noise import factors, round_to_closest_divisor


def test_load_image(example_image):
    t = load_image(str(example_image))
    assert torch.is_tensor(t) and t.ndim == 4 and t.shape[1] == 3
    assert 0 <= t.min() and t.max() <= 1


def test_resample(example_image):
    t = load_image(str(example_image))
    out = resample(t, 32)
    assert out.ndim == 4 and out.shape[1] == 3
    assert max(out.shape[-2:]) < max(t.shape[-2:])


def test_img_tensor_roundtrip(example_image):
    t = load_image(str(example_image))
    t2 = img2tensor(tensor2img(t))
    assert t2.shape == t.shape
    assert (t - t2).abs().max() <= 1 / 255 + 1e-6


def test_write_video(tmp_path, assert_video):
    from maua.ops.video import write_video

    out_file = tmp_path / "out.mp4"
    write_video(tensor=torch.rand(4, 3, 64, 64), output_file=str(out_file), fps=4)
    assert_video(out_file, min_frames=4)


def test_noise_helpers():
    assert set(factors(12).tolist()) == {1, 2, 3, 4, 6, 12}
    assert round_to_closest_divisor(12, 5) in (4, 6)
