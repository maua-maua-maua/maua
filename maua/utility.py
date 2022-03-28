import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


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


def name(s):
    return s.split("/")[-1].split(".")[0]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, output_path):
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    return output_path


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
