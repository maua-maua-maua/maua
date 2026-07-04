from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.download]

PATCHES = Path(__file__).resolve().parents[2] / "maua" / "audiovisual" / "patches" / "examples"


def test_audio2video(short_audio, stylegan2_model, tmp_path, assert_video):
    from maua.audiovisual.generate import generate_audiovisal_from_patch

    out_file = tmp_path / "audioreactive.mp4"
    generate_audiovisal_from_patch(
        audio_file=str(short_audio),
        model_file=str(stylegan2_model),
        patch_file=str(PATCHES / "stylegan2.py"),
        patch_name=None,
        renderer="ffmpeg",
        renderer_kwargs=dict(output_file=str(out_file), ffmpeg_preset="ultrafast"),
        fps=8,
        out_size=(128, 128),
        resize_strategy="pad-zero",
        resize_layer=0,
    )
    assert_video(out_file, min_frames=8)
