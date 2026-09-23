"""Liveness checks for every remote weight host hardcoded in the repo.

Dead weight hosts are the most common silent rot in this codebase (mirror.io.community,
web.eecs.umich.edu, eaidata.bmk.sh all died unnoticed). These tests HEAD/range-probe
each currently-wired host so a dead mirror fails loudly in the smoke tier instead of
mid-run for a user. Network-only — no weights are downloaded (few-KB probes).
"""

import pytest
import requests

pytestmark = [pytest.mark.download]

TIMEOUT = 30
UA = {"User-Agent": "Maua", "From": "https://github.com/maua-maua-maua/maua"}


def probe(url, headers=None):
    r = requests.head(url, allow_redirects=True, timeout=TIMEOUT, headers={**UA, **(headers or {})})
    if r.status_code in (403, 405):  # some hosts reject HEAD; fall back to a 1-byte range GET
        r = requests.get(
            url, allow_redirects=True, timeout=TIMEOUT, headers={**UA, "Range": "bytes=0-0", **(headers or {})}, stream=True
        )
    return r


@pytest.mark.parametrize(
    "url",
    [
        # VQGAN imagenet (parameterizations/vqgan.py) — original CompVis heibox share
        "https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
        "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
        # sflckr (parameterizations/vqgan.py)
        "https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1",
        # pgg vgg16/vgg19 (perceptors/vgg_pgg.py) — HF mirror
        "https://huggingface.co/spaces/AfrodreamsAI/afrodreams/resolve/main/models/vgg16-00b39a1b.pth",
        "https://huggingface.co/spaces/AfrodreamsAI/afrodreams/resolve/main/models/vgg19-d01eb7cb.pth",
    ],
)
def test_weight_host_alive(url):
    r = probe(url)
    assert r.status_code in (200, 206), f"weight host dead ({r.status_code}): {url}"


def test_wikiart_github_release_assets_present():
    """wikiart VQGAN weights come from the Eleiber/VQGAN-Mirrors release via the API asset endpoint."""
    r = requests.get("https://api.github.com/repos/Eleiber/VQGAN-Mirrors/releases", timeout=TIMEOUT, headers=UA)
    r.raise_for_status()
    assets = {a["name"]: a for rel in r.json() for a in rel.get("assets", [])}
    for name, min_size in [
        ("wikiart_1024.yaml", 500),
        ("wikiart_1024.ckpt", 9e8),
        ("wikiart_16384.yaml", 500),
        ("wikiart_16384.ckpt", 9e8),
    ]:
        assert name in assets, f"{name} missing from Eleiber/VQGAN-Mirrors release"
        assert assets[name]["size"] > min_size, f"{name} suspiciously small: {assets[name]['size']}"

    # the API asset endpoint must actually stream binary (browser_download_url serves a login page)
    aid = assets["wikiart_16384.yaml"]["id"]
    r = requests.get(
        f"https://api.github.com/repos/Eleiber/VQGAN-Mirrors/releases/assets/{aid}",
        headers={**UA, "Accept": "application/octet-stream"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200 and r.content.startswith(b"model:"), "API asset endpoint no longer streams content"


def test_maua_weights_hf_repo_alive():
    """Primary re-host: HF wav/maua-weights (Phase B). fetch_model() resolves through here."""
    r = requests.get("https://huggingface.co/api/models/wav/maua-weights/tree/main", timeout=TIMEOUT, headers=UA)
    assert r.status_code == 200, f"HF weights repo unreachable: {r.status_code}"
    assert len(r.json()) > 5, "HF weights repo lost its files?"
