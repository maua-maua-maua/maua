import functools
import pathlib
import random
import shutil
import tarfile
import zipfile

import numpy as np
import requests
import torch
from tqdm.auto import tqdm


def info(x, y=None, label=None):
    if label is None:
        if y is None:
            print(
                f"{x.min().detach().cpu().item():.2f}",
                f"{x.mean().detach().cpu().item():.2f}",
                f"{x.max().detach().cpu().item():.2f}",
                x.shape,
            )
        else:
            print(
                f"{x.min().detach().cpu().item():.2f}",
                f"{x.mean().detach().cpu().item():.2f}",
                f"{x.max().detach().cpu().item():.2f}",
                x.shape,
                f"{y.min().detach().cpu().item():.2f}",
                f"{y.mean().detach().cpu().item():.2f}",
                f"{y.max().detach().cpu().item():.2f}",
                y.shape,
            )
    else:
        if y is None:
            print(
                label,
                f"{x.min().detach().cpu().item():.2f}",
                f"{x.mean().detach().cpu().item():.2f}",
                f"{x.max().detach().cpu().item():.2f}",
                x.shape,
            )
        else:
            print(
                label,
                f"{x.min().detach().cpu().item():.2f}",
                f"{x.mean().detach().cpu().item():.2f}",
                f"{x.max().detach().cpu().item():.2f}",
                x.shape,
                f"{y.min().detach().cpu().item():.2f}",
                f"{y.mean().detach().cpu().item():.2f}",
                f"{y.max().detach().cpu().item():.2f}",
                y.shape,
            )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def name(s):
    return s.split("/")[-1].split(".")[0]


def download(url, filename):
    headers = {"User-Agent": "Maua", "From": "https://github.com/maua-maua-maua/maua"}
    r = requests.get(url, stream=True, allow_redirects=True, headers=headers)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = f"Downloading {filename}" + (" (Unknown total file size)" if file_size == 0 else "")
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path


def fetch(path_or_url):
    if not (path_or_url.startswith("http://") or path_or_url.startswith("https://")):
        return open(path_or_url, "rb")
    return requests.get(path_or_url, stream=True).raw


def unzip(file, path):
    if file.endswith("tar.gz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(path)
        tar.close()
    elif file.endswith("tar"):
        tar = tarfile.open(file, "r:")
        tar.extractall(path)
        tar.close()
    elif file.endswith("zip"):
        zip = zipfile.open(file)
        zip.extractall(path)
        zip.close()


def parse_prompt(prompt):
    if prompt.startswith("http://") or prompt.startswith("https://"):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])
