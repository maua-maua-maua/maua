"""
Single chokepoint for model-weight downloads.

Resolution order:
1. $MAUA_MODELZOO (default: ./modelzoo) — if the file is already there, use it
2. the Hugging Face Hub repo (default: wav/maua-weights, where maua's weights are re-hosted)
3. the original URL, if one is still known

All files end up as plain files under the modelzoo directory, so code that builds
"modelzoo/<name>" paths keeps working.
"""

import os
from pathlib import Path

HF_WEIGHTS_REPO = "wav/maua-weights"


def modelzoo_dir() -> Path:
    path = Path(os.environ.get("MAUA_MODELZOO", "modelzoo"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_model(filename, url=None, hf_repo=HF_WEIGHTS_REPO, hf_filename=None) -> str:
    """Return a local path to a weight file, downloading it into the modelzoo if necessary."""
    path = modelzoo_dir() / filename
    if path.exists():
        return str(path)

    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=hf_repo, filename=hf_filename or filename, local_dir=modelzoo_dir())
        if (modelzoo_dir() / (hf_filename or filename)).exists() and not path.exists():
            os.rename(modelzoo_dir() / (hf_filename or filename), path)
        return str(path)
    except Exception as e:
        if url is None:
            raise
        print(f"Download of {filename} from Hugging Face Hub ({hf_repo}) failed ({e}), trying {url}...")

    from maua.utility import download

    download(url, str(path))
    return str(path)


def fetch_folder(dirname, hf_repo=HF_WEIGHTS_REPO) -> str:
    """Return a local path to a directory of weight files, downloading it from the Hub if necessary."""
    path = modelzoo_dir() / dirname
    if not path.exists():
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=hf_repo, allow_patterns=[f"{dirname}/*"], local_dir=modelzoo_dir())
    return str(path)
