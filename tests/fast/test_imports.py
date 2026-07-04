"""Every core module must import cleanly.

XFAIL_IMPORTS is the living triage board: modules known to be broken get an entry
with the reason (and, once quarantined, a pointer to maua/legacy). Remove entries
as they are repaired.
"""

import importlib
from pathlib import Path

import pytest

MAUA = Path(__file__).resolve().parents[2] / "maua"

# Third-party submodules and script-only entry points are not part of the import surface.
EXCLUDE = (
    "submodules",
    "GAN/nv",
    "GAN/studio",
    "GAN/pix2pix",
    "audio",
)

XFAIL_IMPORTS: dict[str, str] = {
    # module: reason
    "maua.diffusion.flux2hd": "loads the full FLUX pipeline (multi-GB download) at import time; repair in Phase B",
    # --- missing pix2pix submodule (gitlink was never committed; re-add with:
    #     git submodule add https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix maua/GAN/pix2pix)
    "maua.GAN.ZSSGAN.model.ZSSGAN": "imports maua.GAN.pix2pix, a submodule missing from the git index",
    "maua.GAN.nada": "imports ZSSGAN, which needs the missing maua.GAN.pix2pix submodule",
    # --- Colab-notebook pastes: model loads and hardcoded /home/hans paths at import time
    "maua.GAN.icgan.generate": "Colab-style script: loads SwAV/IC-GAN weights and loops over hardcoded datasets at import; legacy candidate",
    "maua.GAN.icgan.guided": "star-imports maua.GAN.icgan.generate (see above)",
    "maua.style.omnimae": "loads modelzoo/vitl_ssv2_ft.torch at import time; legacy candidate",
    # --- heavy/abandoned optional deps for the GAN training stack
    "maua.GAN.training.trainer": "requires ffcv (unmaintained, compile-heavy); legacy candidate",
    "maua.GAN.training.train_v0": "requires padl (abandoned); legacy candidate",
    "maua.GAN.training.dataset.image": "requires ffcv (unmaintained, compile-heavy); legacy candidate",
    "maua.GAN.training.models.experimental.deepinvolutional": "requires involution (research dep, not on PyPI)",
    "maua.GAN.training.models.experimental.equivariant": "requires escnn (research dep with unsatisfiable pins)",
    "maua.GAN.training.models.experimental.stylehypermixerfly": "requires torch_butterfly (research dep)",
    # --- selfsupervised audioreactive research extras
    "maua.audiovisual.audioreactive.selfsupervised.features.correlation": "requires anatome (unsatisfiable pins on modern torch)",
    "maua.audiovisual.audioreactive.selfsupervised.features.efficient_quantile": "C++ extension that must be built in-place (see its setup.py)",
    "maua.audiovisual.audioreactive.selfsupervised.features.efficient_quantile.setup": "setuptools build script, not an importable module",
    # --- autoregressive models broken by py3.12 / new huggingface_hub / old protobuf
    "maua.autoregressive.min_dalle.generate": "minDALL-E submodule uses mutable dataclass defaults (rejected by python>=3.12)",
    "maua.autoregressive.rq_dalle": "rqvae submodule uses mutable dataclass defaults (rejected by python>=3.12)",
    "maua.autoregressive.ru_dalle": "rudalle package needs huggingface_hub.cached_download (removed)",
    "maua.autoregressive.ru_dalle.api": "rudalle package needs huggingface_hub.cached_download (removed)",
    "maua.autoregressive.ru_dalle.finetune": "rudalle package needs huggingface_hub.cached_download (removed)",
    "maua.autoregressive.ru_dalle.generate": "rudalle package needs huggingface_hub.cached_download (removed)",
    "maua.autoregressive.cog.video.generate": "CogVideo/icetk need protobuf<3.20 era APIs; legacy candidate",
    "maua.autoregressive.cog.video.infinite": "CogVideo/icetk need protobuf<3.20 era APIs; legacy candidate",
}

# No single module should take longer than this to import; catches import-time downloads/compiles.
pytestmark = pytest.mark.timeout(180)


def discover_modules():
    modules = []
    for f in sorted(MAUA.rglob("*.py")):
        rel = f.relative_to(MAUA)
        s = str(rel)
        if any(s == e or s.startswith(e + "/") for e in EXCLUDE) or "__pycache__" in s:
            continue
        if f.name == "__main__.py":
            continue  # importing a __main__ executes it
        name = "maua." + ".".join(rel.with_suffix("").parts)
        name = name.removesuffix(".__init__")
        modules.append(name)
    return modules


@pytest.mark.parametrize("module", discover_modules())
def test_import(module):
    if module in XFAIL_IMPORTS:
        pytest.xfail(XFAIL_IMPORTS[module])
    importlib.import_module(module)
