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
