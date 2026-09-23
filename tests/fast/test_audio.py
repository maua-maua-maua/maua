"""CPU-fast behavioral coverage for the rescued maua/audio tree.

The audio modules were rescued into the import surface post-Phase-B; these tests pin
the numerically-sensitive pieces (PQMF filterbank math survived the scipy>=1.15 API
migration) so a regression can't hide behind a clean import.
"""

import numpy as np
import torch


def test_kaiser_filter_dc_gain():
    from maua.audio.rave.rave.pqmf import kaiser_filter

    h = kaiser_filter(0.5, 100)
    assert np.isclose(np.sum(h), 1.0, atol=1e-4), "lowpass prototype must have ~unit DC gain"


def test_get_prototype_optimizes_within_range():
    """fmin explores out-of-range cutoffs; the loss guard must keep firwin from raising."""
    from maua.audio.rave.rave.pqmf import get_prototype

    p = get_prototype(100, 4, 33)
    assert p.shape == (33,)
    assert np.all(np.isfinite(p))


def test_pqmf_roundtrip():
    """Forward/inverse polyphase filterbank must reconstruct the signal."""
    from maua.audio.rave.rave.pqmf import PQMF

    torch.manual_seed(0)
    pq = PQMF(70, 4)
    x = torch.randn(1, 1, 2**12)
    z = pq(x)
    assert z.shape == (1, 4, 2**12 // 4), "analysis must split into 4 critically-sampled bands"
    y = pq.inverse(z)
    assert y.shape == x.shape
    err = (x - y).abs().mean().item()
    assert err < 0.01, f"PQMF reconstruction error too high: {err}"


def test_pqmf_two_band():
    from maua.audio.rave.rave.pqmf import PQMF

    torch.manual_seed(0)
    pq = PQMF(70, 2)
    x = torch.randn(1, 1, 2**11)
    err = (x - pq.inverse(pq(x))).abs().mean().item()
    assert err < 0.01


def test_audioreactive_rave_shim_paths():
    """The audio package shims must expose the vendored top-level packages."""
    import maua.audio  # noqa: F401  (installs sys.path shim)
    import maua.audio.rave  # noqa: F401

    import jukebox  # bare vendored names resolve through the shim
    import rave
    import prior
    import waveshaping

    for mod in (jukebox, rave, prior, waveshaping):
        assert "maua/audio" in mod.__path__[0], f"{mod.__name__} must resolve to the vendored copy"
