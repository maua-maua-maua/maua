from typing import List

import torch

from ..ops.image import luminance
from . import mm, sniklaus


def get_flow_model(
    which: List[str] = [
        # "unflow",
        # "pwc",
        # "spynet",
        # "liteflownet",
        # "gma/gma_plus-p_8x2_120k_mixed_368x768",
        # "raft/raft_8x2_100k_mixed_368x768",
        "farneback",
    ]
):
    pred_fns = []

    if "unflow" in which:
        pred_fns.append(sniklaus.get_prediction_fn("unflow"))
    if "pwc" in which:
        pred_fns.append(sniklaus.get_prediction_fn("pwc"))
    if "spynet" in which:
        pred_fns.append(sniklaus.get_prediction_fn("spynet"))
    if "liteflownet" in which:
        pred_fns.append(sniklaus.get_prediction_fn("liteflownet"))

    for w in which:
        if w in mm.AVAILABLE_MODELS:
            pred_fns.append(mm.get_prediction_fn(w))

    if "farneback" in which:
        import cv2

        pred_fns.append(
            lambda im1, im2: torch.from_numpy(
                cv2.calcOpticalFlowFarneback(
                    luminance(im1.detach().squeeze().permute(1, 2, 0)).mul(255).byte().cpu().numpy(),
                    luminance(im2.detach().squeeze().permute(1, 2, 0)).mul(255).byte().cpu().numpy(),
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
            .unsqueeze(0)
            .to(im1.device)
        )

    if "deepflow2" in which:
        raise Exception("deepflow2 not working quite yet...")
        from thoth.deepflow2 import deepflow2
        from thoth.deepmatching import deepmatching

        models.append(lambda im1, im2: deepflow2(im1, im2, deepmatching(im1, im2)))

    return lambda im1, im2: torch.mean(torch.stack([pred(im1, im2) for pred in pred_fns]), dim=0).to(im1).float()


from .consistency import check_consistency, check_consistency_np
from .lib import flow_warp_map, get_consistency_map, preprocess_optical_flow
from .utils import flow_to_image, read_flow, resample_flow, write_flow
