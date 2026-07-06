import os
import sys

import torch
import torch.nn.functional as F
from resize_right import resize

# The pwc / liteflownet / unflow backends need a custom CUDA correlation kernel compiled at runtime
# via cupy. Install a cupy build matching the CUDA toolchain (e.g. `pip install cupy-cuda12x`); keep
# it <14 so it stays compatible with numpy 1.26. The vendored correlation.py files were migrated off
# the removed `cupy.cuda.compile_with_cache` API to `cupy.RawModule`. spynet is pure torch (no cupy).

# remove shape asserts from optical flow files
for file in [
    os.path.dirname(__file__) + "/../submodules/unflow/run.py",
    os.path.dirname(__file__) + "/../submodules/pwc/run.py",
    os.path.dirname(__file__) + "/../submodules/spynet/run.py",
    os.path.dirname(__file__) + "/../submodules/liteflownet/run.py",
]:
    with open(file, "r") as f:
        txt = f.read().replace("assert", "# assert").replace("# #", "#")
    with open(file, "w") as f:
        f.write(txt)

# cupy removed `cupy.cuda.compile_with_cache` in v13; migrate the vendored correlation kernels to
# `cupy.RawModule` in place (idempotent) so the neural backends keep working across submodule resets.
for file in [
    os.path.dirname(__file__) + "/../submodules/unflow/correlation/correlation.py",
    os.path.dirname(__file__) + "/../submodules/pwc/correlation/correlation.py",
    os.path.dirname(__file__) + "/../submodules/liteflownet/correlation/correlation.py",
]:
    with open(file, "r") as f:
        txt = f.read()
    patched = txt.replace(
        "cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)",
        "cupy.RawModule(code=strKernel).get_function(strFunction)",
    )
    if patched != txt:
        with open(file, "w") as f:
            f.write(patched)


def preprocess(im, h, w):
    if h is not None and w is not None:
        im = resize(im, out_shape=(h, w))
    im = im[:, [2, 1, 0]]  # RGB -> BGR
    return im.float().squeeze()


@torch.no_grad()
def predict(model, im1, im2, flowh=None, floww=None):
    b, c, h, w = im1.shape
    tens1 = preprocess(im1, flowh, floww)
    tens2 = preprocess(im2, flowh, floww)
    model_out = model(tens1, tens2).unsqueeze(0)
    output = F.interpolate(input=model_out, size=(h, w), mode="bilinear", align_corners=False)
    return output.permute(0, 2, 3, 1)


def get_prediction_fn(which, use_training_size=False):
    if "unflow" == which:
        del sys.argv[1:]
        from maua.submodules.unflow.run import estimate as unflow

        torch.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away

        if use_training_size:
            size = (384, 1280)
        else:
            size = (None, None)

        return lambda im1, im2: predict(unflow, im1, im2, *size).to(im1.device)

    if "pwc" == which:
        del sys.argv[1:]
        from maua.submodules.pwc.run import estimate as pwc

        torch.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away

        if use_training_size:
            size = (436, 1024)
        else:
            size = (None, None)

        return lambda im1, im2: predict(pwc, im1, im2, *size).to(im1.device)

    if "spynet" == which:
        del sys.argv[1:]
        from maua.submodules.spynet.run import estimate as spynet

        torch.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away

        if use_training_size:
            size = (416, 1024)
        else:
            size = (None, None)

        return lambda im1, im2: predict(spynet, im1, im2, *size).to(im1.device)

    if "liteflownet" == which:
        del sys.argv[1:]
        from maua.submodules.liteflownet.run import estimate as liteflownet

        torch.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away

        if use_training_size:
            size = (436, 1024)
        else:
            size = (None, None)

        return lambda im1, im2: predict(liteflownet, im1, im2, *size).to(im1.device)
