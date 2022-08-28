import os
import sys
from glob import glob
from importlib.util import module_from_spec, spec_from_file_location
from typing import Generator
from zipfile import ZipFile

import gdown
import torch
from decord import VideoReader
from torch.nn import functional as F
from tqdm import tqdm

from ....utility import download

URLS = {
    "RIFE-1.0": "1U2AGFY00hafsPmm94-6deeM-9feGN-qg",
    "RIFE-1.1": "1wsQIhHZ3Eg4_AfCXItFKqqyDMB4NS0Yd",
    "RIFE-2.0": "https://share.marqt40.com/index.php/s/TDDBdKGX6ZzaLXi/download/RIFE_trained_model_v2.0_HDv2.zip",
    "RIFE-2.1": "https://share.marqt40.com/index.php/s/mAGy78zzNpTAXWo/download/RIFE_trained_model_v2.1_HDv2.zip",
    "RIFE-2.2": "https://share.marqt40.com/index.php/s/qF4Dgf5XAtd48yN/download/RIFE_trained_model_v2.2_HDv2.zip",
    "RIFE-2.3": "https://share.marqt40.com/index.php/s/5xbsPzn9K4oRGAp/download/RIFE_trained_model_v2.3_HDv2.zip",
    "RIFE-2.4": "https://share.marqt40.com/index.php/s/PcQrq4ByZrsPXaR/download/RIFE_trained_model_v2.4_HDv2.zip",
    "RIFE-3.0": "1JmwH8L3pdy49NroCVwracDW5UM43AAqd",
    "RIFE-3.1": "1xn4R3TQyFhtMXN2pa3lRB8cd4E1zckQe",
    "RIFE-3.2": "13x77FcKwqoCv8ZalPP7f95G3UJNYCJov",
    "RIFE-3.4": "10-2AaFUyX-c7yCfubsxF2NTvM7DgvS8l",
    "RIFE-3.5": "1YEi5KAdo0e6XnCTcbzOGCNtU33Lc2yO2",
    "RIFE-3.6": "1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_",
    "RIFE-3.8": "1O5KfS3KzZCY3imeCr2LCsntLhutKuAqj",
    "RIFE-3.9": "1iosmPTt2ayAdSMqnI1cxO_R1-Qhrranp",
    "RIFE-4.0": "1mUK9iON6Es14oK46-cCflRoPTeGiI_A9",
}
VERSIONS = [ver.replace("RIFE-", "") for ver in URLS.keys()]


def load_model(model_name="RIFE-2.3", device="cuda", fp16=False):

    if fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    version = model_name.replace("RIFE-", "")
    model_dir = f"modelzoo/RIFE_HDv{version}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

        if version.startswith("2"):
            download(URLS[model_name], f"{model_dir}.zip")
        else:
            gdown.download(id=URLS[model_name], output=f"{model_dir}.zip")

        with ZipFile(f"{model_dir}.zip", "r") as archive:
            for info in archive.infolist():
                if info.filename[-1] not in ["/", ".DS_Store", "._.DS_Store", "__pycache__"]:
                    info.filename = os.path.basename(info.filename)
                    archive.extract(info, model_dir)
        os.remove(f"{model_dir}.zip")

    sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../../../submodules/RIFE/")
    if version.startswith("1"):
        from ....submodules.RIFE.model.oldmodel.RIFE_HD import Model
    elif version.startswith("2"):
        from ....submodules.RIFE.model.oldmodel.RIFE_HDv2 import Model
    else:
        for file in glob(f"{model_dir}/*.py"):
            with open(file, "r") as f:
                txt = f.read().replace("train_log.", "")
            with open(file, "w") as f:
                f.write(txt)
        sys.path.append(model_dir)
        spec = spec_from_file_location("RIFE_HDv3", f"{model_dir}/RIFE_HDv3.py")
        RIFE_HDv3 = module_from_spec(spec)
        spec.loader.exec_module(RIFE_HDv3)
        Model = RIFE_HDv3.Model

    model = Model()
    model.load_model(model_dir, -1)
    model.eval()
    model.device()
    model.device = device

    if fp16:
        torch.set_default_tensor_type(torch.FloatTensor)

    return model


def recursive_inference(model, I0, I1, n):
    middle = model.inference(I0, I1, 1)
    if n == 1:
        return [middle]
    first_half = recursive_inference(model, I0, middle, n=n // 2)
    second_half = recursive_inference(model, middle, I1, n=n // 2)
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]


@torch.inference_mode()
def interpolate(frame1, frame2, model, factor=2, fp16=False):
    if fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    b, c, h, w = frame1.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)

    if fp16:
        frame1, frame2 = frame1.half(), frame2.half()
    frame1, frame2 = F.pad(frame1, padding), F.pad(frame2, padding)

    intermediate_frames = recursive_inference(model, frame1, frame2, factor)
    for frame in intermediate_frames:
        yield frame[..., :h, :w].float()

    if fp16:
        torch.set_default_tensor_type(torch.FloatTensor)
