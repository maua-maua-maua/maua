import os
import subprocess
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "workspace" / "examples"


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="requires a CUDA GPU")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _torch_global_state():
    """Restore torch's process-global default device/dtype after every test.

    Several code paths legitimately flip global torch state during a run
    (`nca/{train,generate}.py` call `torch.set_default_device('cuda')`;
    `super/video/framerate/rife.py` uses `set_default_tensor_type`). Their
    "restore" sets the state back to the *value* they read, but on a fresh
    process that installs a default-device *mode* that wasn't there before,
    which then leaks into later tests (e.g. a cpu default-device mode made
    `flow_warp_map`'s cached grid land on cpu and broke video_diffusion).
    Snapshot here and hard-reset via `set_default_device(None)` so each test
    starts from the pristine no-mode state.
    """
    dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_device(None)
        torch.set_default_dtype(dtype)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def modelzoo():
    path = Path(os.environ.get("MAUA_MODELZOO", REPO / "modelzoo"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def tiny():
    return dict(size=64, steps=2, frames=3, fps=4)


@pytest.fixture(scope="session")
def example_image():
    return EXAMPLES / "image" / "small1.jpg"


@pytest.fixture(scope="session")
def example_image2():
    return EXAMPLES / "image" / "small2.jpg"


@pytest.fixture(scope="session")
def example_video():
    return EXAMPLES / "video" / "small.mp4"


@pytest.fixture(scope="session")
def example_audio():
    return EXAMPLES / "audio" / "Wavefunk - Tau Ceti Alpha.mp3"


@pytest.fixture(scope="session")
def short_video(tmp_path_factory, example_video):
    """A 6-frame, 128px clip so per-frame methods stay fast."""
    out = tmp_path_factory.mktemp("clips") / "short.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(example_video), "-t", "0.5", "-vf", "scale=128:128", str(out)],
        check=True,
    )
    return out


@pytest.fixture(scope="session")
def short_audio(tmp_path_factory, example_audio):
    """A 2-second audio clip so audio-reactive methods stay fast."""
    out = tmp_path_factory.mktemp("clips") / "short.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(example_audio), "-t", "2", str(out)], check=True)
    return out


@pytest.fixture
def small_image_64(tmp_path, example_image):
    """The example image downscaled to 64px, for upscaling tests."""
    from PIL import Image

    out = tmp_path / "small64.png"
    Image.open(example_image).convert("RGB").resize((64, 64)).save(out)
    return out


@pytest.fixture(scope="session")
def stylegan2_model(modelzoo):
    """A StyleGAN2 checkpoint: MAUA_STYLEGAN2_PKL, any .pkl already in the modelzoo, or NVIDIA's FFHQ-256."""
    env = os.environ.get("MAUA_STYLEGAN2_PKL")
    if env:
        return Path(env)
    # Only reuse a local pkl if it looks like a StyleGAN checkpoint; the model zoo also
    # holds unrelated pickles (e.g. jax_diffusion_*.pkl) that the GAN loaders can't read.
    existing = [p for p in sorted(modelzoo.glob("*.pkl")) if "diffusion" not in p.name.lower()]
    if existing:
        return existing[0]
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl"
    target = modelzoo / "stylegan2-ffhq-256x256.pkl"
    torch.hub.download_url_to_file(url, str(target))
    return target


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path


@pytest.fixture(scope="session")
def assert_image():
    def _assert_image(path, min_size=8):
        from PIL import Image

        path = Path(path)
        assert path.exists() and path.stat().st_size > 0, f"{path} missing or empty"
        img = Image.open(path)
        assert min(img.size) >= min_size, f"{path} is {img.size}, expected at least {min_size}px"

    return _assert_image


@pytest.fixture(scope="session")
def assert_video():
    def _assert_video(path, min_frames=1):
        import ffmpeg

        path = Path(path)
        assert path.exists() and path.stat().st_size > 0, f"{path} missing or empty"
        info = ffmpeg.probe(str(path))
        stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        n_frames = int(stream.get("nb_frames", 0)) or round(
            float(info["format"]["duration"]) * eval(stream["r_frame_rate"])
        )
        assert n_frames >= min_frames, f"{path} has {n_frames} frames, expected at least {min_frames}"

    return _assert_video
