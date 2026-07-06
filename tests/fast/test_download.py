"""Weight-resolution logic for maua.ops.download — no network required.

The download helpers are the single choke point every model loader goes through, so
their resolution order ($MAUA_MODELZOO hit → Hugging Face → URL fallback) is worth
pinning without actually hitting the network.
"""

from pathlib import Path

import pytest


def test_modelzoo_dir_honors_env(tmp_path, monkeypatch):
    from maua.ops import download

    monkeypatch.setenv("MAUA_MODELZOO", str(tmp_path / "zoo"))
    d = download.modelzoo_dir()
    assert d == tmp_path / "zoo" and d.is_dir(), "modelzoo_dir must create and return the $MAUA_MODELZOO path"


def test_fetch_model_uses_local_file_without_network(tmp_path, monkeypatch):
    from maua.ops import download

    monkeypatch.setenv("MAUA_MODELZOO", str(tmp_path))
    (tmp_path / "weights.pt").write_bytes(b"stub")

    def _boom(*a, **k):  # any network attempt is a bug when the file is already local
        raise AssertionError("fetch_model hit the network despite a local file")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom, raising=False)
    assert download.fetch_model("weights.pt") == str(tmp_path / "weights.pt")


def test_fetch_model_falls_back_to_url(tmp_path, monkeypatch):
    from maua.ops import download

    monkeypatch.setenv("MAUA_MODELZOO", str(tmp_path))

    def _hf_fail(*a, **k):
        raise RuntimeError("no hub")

    calls = {}

    def _fake_download(url, dest):
        calls["url"], calls["dest"] = url, dest
        Path(dest).write_bytes(b"downloaded")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _hf_fail, raising=False)
    monkeypatch.setattr("maua.utility.download", _fake_download, raising=False)

    out = download.fetch_model("weights.pt", url="https://example.com/weights.pt")
    assert out == str(tmp_path / "weights.pt")
    assert calls["url"] == "https://example.com/weights.pt", "URL fallback must be used when the Hub fails"


def test_fetch_model_reraises_when_no_url(tmp_path, monkeypatch):
    from maua.ops import download

    monkeypatch.setenv("MAUA_MODELZOO", str(tmp_path))
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no hub")), raising=False
    )
    with pytest.raises(RuntimeError):
        download.fetch_model("missing.pt")
