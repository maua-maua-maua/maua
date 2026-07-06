from typing import List

import torch

from maua.flow import mm, sniklaus
from maua.ops.image import luminance


def get_flow_model(
    which: list[str] = [
        # "unflow",
        # "pwc",
        # "spynet",
        # "liteflownet",
        # "gma/gma_plus-p_8x2_120k_mixed_368x768",
        # "raft/raft_8x2_100k_mixed_368x768",
        "farneback",
    ],
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
            lambda im1, im2: (
                torch
                .from_numpy(
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
        )

    if "deepflow2" in which:
        raise Exception("deepflow2 not working quite yet...")
        from thoth.deepflow2 import deepflow2
        from thoth.deepmatching import deepmatching

        pred_fns.append(lambda im1, im2: deepflow2(im1, im2, deepmatching(im1, im2)))

    return lambda im1, im2: torch.mean(torch.stack([pred(im1, im2) for pred in pred_fns]), dim=0).to(im1).float()


from maua.flow.consistency import check_consistency as check_consistency
from maua.flow.consistency import check_consistency_np as check_consistency_np
from maua.flow.lib import (
    flow_warp_map as flow_warp_map,
)
from maua.flow.lib import (
    get_consistency_map as get_consistency_map,
)
from maua.flow.lib import (
    preprocess_optical_flow as preprocess_optical_flow,
)
from maua.flow.utils import (
    flow_to_image as flow_to_image,
)
from maua.flow.utils import (
    read_flow as read_flow,
)
from maua.flow.utils import (
    resample_flow as resample_flow,
)
from maua.flow.utils import (
    write_flow as write_flow,
)
