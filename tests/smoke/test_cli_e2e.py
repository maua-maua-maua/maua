"""End-to-end CLI smoke: `python -m maua diffusion image` must run and write a real PNG.

The fast tier proves the parser binds a dispatch function; this proves the dispatch actually
executes the pipeline through a fresh process (argv parsing → lazy import → main → saved file).
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.download, pytest.mark.backend_stable]

REPO = Path(__file__).resolve().parents[2]


def test_diffusion_image_cli(tmp_path, assert_image):
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable, "-m", "maua", "diffusion", "image",
        "--text", "a colorful painting of a forest",
        "--sizes", "64,64",
        "--timesteps", "2",
        "--diffusion", "stable",
        "--sampler", "euler",
        "--out-dir", str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-3000:]}"
    pngs = list(out_dir.glob("*.png"))
    assert pngs, f"no image written to {out_dir}"
    assert_image(pngs[0], min_size=64)
