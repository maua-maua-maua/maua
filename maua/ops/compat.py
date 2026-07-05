"""
Shims for modules that old third-party dependencies (submodules, basicsr, ic_gan, ...) import
from locations that no longer exist in current torchvision. Installed by maua/__init__.py so any
entrypoint into the package gets them.
"""

import sys
import types


def install_shims():
    import torch
    import torch.hub
    import torchvision.models
    import torchvision.transforms.functional

    # torch._six was removed in torch 1.9+; the VQGAN/taming submodule still imports
    # string_classes/int_classes/container_abcs from it.
    if "torch._six" not in sys.modules:
        import collections.abc

        mod = types.ModuleType("torch._six")
        mod.string_classes = (str, bytes)
        mod.int_classes = (int,)
        mod.container_abcs = collections.abc
        mod.inf = float("inf")
        sys.modules["torch._six"] = mod
        torch._six = mod

    # torchvision.models.utils.load_state_dict_from_url moved to torch.hub (torchvision 0.13)
    if not hasattr(torchvision.models, "utils"):
        mod = types.ModuleType("torchvision.models.utils")
        mod.load_state_dict_from_url = torch.hub.load_state_dict_from_url
        torchvision.models.utils = mod
        sys.modules["torchvision.models.utils"] = mod

    # torchvision.transforms.functional_tensor was removed in torchvision 0.17 (basicsr needs it)
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        try:
            import torchvision.transforms.functional_tensor  # noqa: F401
        except ImportError:
            mod = types.ModuleType("torchvision.transforms.functional_tensor")
            mod.rgb_to_grayscale = torchvision.transforms.functional.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = mod

    # huggingface_hub.cached_download was removed (~0.26); the ru_dalle submodule still does
    # `from huggingface_hub import hf_hub_url, cached_download` and calls
    # cached_download(url, cache_dir=..., force_filename=...). It only ever fetches a plain URL
    # (built via hf_hub_url) into cache_dir/force_filename, so reimplement that minimal contract.
    import huggingface_hub

    if not hasattr(huggingface_hub, "cached_download"):

        def cached_download(url, cache_dir=None, force_filename=None, **kwargs):
            import hashlib
            import os

            import requests

            cache_dir = cache_dir or os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "hub"
            )
            os.makedirs(cache_dir, exist_ok=True)
            filename = force_filename or hashlib.sha256(url.encode()).hexdigest()
            dest = os.path.join(cache_dir, filename)
            if not os.path.exists(dest):
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    tmp = dest + ".incomplete"
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                    os.replace(tmp, dest)
            return dest

        huggingface_hub.cached_download = cached_download

    # icetk (CogVideo's tokenizer) ships a sentencepiece_model_pb2 stub generated against an ancient
    # protobuf that the modern runtime rejects ("Descriptors cannot be created directly"). The
    # `sentencepiece` package ships an up-to-date, API-compatible equivalent (same .proto, exposes
    # ModelProto) — pre-inject it as icetk's submodule so icetk never loads its stale file.
    if "icetk.sentencepiece_model_pb2" not in sys.modules:
        try:
            import sentencepiece.sentencepiece_model_pb2 as _sp_model_pb2

            sys.modules["icetk.sentencepiece_model_pb2"] = _sp_model_pb2
        except ImportError:
            pass

    # SwissArmyTransformer (used by the CogVideo submodule) was renamed to the import name `sat`
    # on PyPI (>=0.4); the CogVideo code still does `from SwissArmyTransformer... import ...`.
    # Alias the old top-level name to the installed `sat` package so its submodules resolve.
    if "SwissArmyTransformer" not in sys.modules:
        try:
            import SwissArmyTransformer  # noqa: F401
        except ImportError:
            try:
                import sat

                sys.modules["SwissArmyTransformer"] = sat
            except ImportError:
                pass
