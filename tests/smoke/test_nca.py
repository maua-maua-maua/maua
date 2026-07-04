import pytest

pytestmark = [pytest.mark.gpu]


def test_nca_train_and_generate(example_image, tmp_path, assert_video):
    from maua.nca.generate import generate
    from maua.nca.train import train

    checkpoint = train(style_file=str(example_image), out_dir=str(tmp_path), n_steps=3)

    output = generate(checkpoint, str(tmp_path / "nca.mp4"), num_frames=5, size=64)
    assert_video(output, min_frames=3)
