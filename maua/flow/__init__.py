import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from maua.ops.image import luminance, resample

# remove shape asserts from optical flow files
for file in [
    "submodules/unflow/run.py",
    "submodules/pwc/run.py",
    "submodules/spynet/run.py",
    "submodules/liteflownet/run.py",
]:
    with open(file, "r") as f:
        txt = f.read().replace("assert", "# assert").replace("# #", "#")
    with open(file, "w") as f:
        f.write(txt)


def preprocess(im, h, w):
    im = im.permute(2, 0, 1).unsqueeze(0)
    if h is not None and w is not None:
        im = resample(im, (h, w))
    im = im[:, [2, 1, 0]]  # RGB -> BGR
    return im.float().squeeze()


def predict(model, im1, im2, flowh=None, floww=None):
    h, w, _ = im1.shape
    tens1 = preprocess(im1, flowh, floww)
    tens2 = preprocess(im2, flowh, floww)
    model_out = model(tens1, tens2).unsqueeze(0)
    output = F.interpolate(input=model_out, size=(h, w), mode="bilinear", align_corners=False)
    return output.squeeze().permute(1, 2, 0).cpu().numpy()


def get_flow_model(which: List[str] = ["pwc", "spynet", "liteflownet"], use_training_size=False):
    pred_fns = []

    if "unflow" in which:
        del sys.argv[1:]
        from submodules.unflow.run import estimate as unflow

        del sys.path[-1]
        torch.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away

        if use_training_size:
            size = (384, 1280)
        else:
            size = (None, None)

        pred_fns.append(lambda im1, im2: predict(unflow, im1, im2, *size))

    if "pwc" in which:
        del sys.argv[1:]
        from submodules.pwc.run import estimate as pwc

        del sys.path[-1]
        torch.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away

        if use_training_size:
            size = (436, 1024)
        else:
            size = (None, None)

        pred_fns.append(lambda im1, im2: predict(pwc, im1, im2, *size))

    if "spynet" in which:
        del sys.argv[1:]
        from submodules.spynet.run import estimate as spynet

        torch.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away

        if use_training_size:
            size = (416, 1024)
        else:
            size = (None, None)

        pred_fns.append(lambda im1, im2: predict(spynet, im1, im2, *size))

    if "liteflownet" in which:
        del sys.argv[1:]
        from submodules.liteflownet.run import estimate as liteflownet

        del sys.path[-1]
        torch.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away

        if use_training_size:
            size = (436, 1024)
        else:
            size = (None, None)

        pred_fns.append(lambda im1, im2: predict(liteflownet, im1, im2, *size))

    if "farneback" in which:
        import cv2

        pred_fns.append(
            lambda im1, im2: cv2.calcOpticalFlowFarneback(
                luminance(im1).mul(255).numpy().astype(np.uint8),
                luminance(im2).mul(255).numpy().astype(np.uint8),
                flow=None,
                pyr_scale=0.8,
                levels=15,
                winsize=15,
                iterations=15,
                poly_n=7,
                poly_sigma=1.5,
                flags=10,
            )
        )

    if "deepflow2" in which:
        raise Exception("deepflow2 not working quite yet...")
        from thoth.deepflow2 import deepflow2
        from thoth.deepmatching import deepmatching

        models.append(lambda im1, im2: deepflow2(im1, im2, deepmatching(im1, im2)))

    return lambda im1, im2: np.sum(pred(im1, im2) for pred in pred_fns) / len(pred_fns)


from .consistency import check_consistency, motion_edge
from .utils import resample_flow, read_flow, write_flow, flow_to_image
