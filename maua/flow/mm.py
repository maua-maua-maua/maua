import os

import torch

from ..utility import download

AVAILABLE_MODELS = [
    "flownet/flownetc_8x1_sfine_sintel_384x448",
    "flownet/flownetc_8x1_slong_flyingchairs_384x448",
    "flownet/flownets_8x1_sfine_sintel_384x448",
    "flownet/flownets_8x1_slong_flyingchairs_384x448",
    "flownet/flownetc_8x1_sfine_flyingthings3d_subset_384x768",
    "flownet2/flownet2_8x1_sfine_flyingthings3d_subset_384x768",
    "flownet2/flownet2css_8x1_sfine_flyingthings3d_subset_384x768",
    "flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448",
    "flownet2/flownet2cs_8x1_sfine_flyingthings3d_subset_384x768",
    "flownet2/flownet2cs_8x1_slong_flyingchairs_384x448",
    "flownet2/flownet2sd_8x1_slong_chairssdhom_384x448",
    "flownet2/flownet2css_8x1_slong_flyingchairs_384x448",
    "gma/gma_plus-p_8x2_50k_kitti2015_288x960",
    "gma/gma_8x2_50k_kitti2015_288x960",
    "gma/gma_8x2_120k_flyingchairs_368x496",
    "gma/gma_p-only_8x2_120k_flyingchairs_368x496",
    "gma/gma_p-only_8x2_120k_mixed_368x768",  # ****************
    "gma/gma_plus-p_8x2_120k_flyingchairs_368x496",
    "gma/gma_plus-p_8x2_120k_flyingthings3d_400x720",
    "gma/gma_p-only_8x2_50k_kitti2015_288x960",
    "gma/gma_p-only_8x2_120k_flyingthings3d_400x720",
    "gma/gma_plus-p_8x2_120k_mixed_368x768",  # ****************
    "gma/gma_8x2_120k_mixed_368x768",  # ****************
    "gma/gma_8x2_120k_flyingthings3d_sintel_368x768",
    "gma/gma_8x2_120k_flyingthings3d_400x720",
    "irr/irrpwc_ft_4x1_300k_sintel_384x768",
    "irr/irrpwc_8x1_sshort_flyingchairsocc_384x448",
    "irr/irrpwc_ft_4x1_300k_kitti_320x896",
    "irr/irrpwc_ft_4x1_300k_sintel_final_384x768",
    "irr/irrpwc_8x1_sfine_half_flyingthings3d_subset_384x768",
    "liteflownet/liteflownet_ft_4x1_500k_kitti_320x896",
    "liteflownet/liteflownet_pre_M4S4R4_8x1_flyingchairs_320x448",
    "liteflownet/liteflownet_pre_M6S6R6_8x1_flyingchairs_320x448",
    "liteflownet/liteflownet_pre_M6S6_8x1_flyingchairs_320x448",
    "liteflownet/liteflownet_8x1_500k_flyingthings3d_subset_384x768",
    "liteflownet/liteflownet_pre_M2S2R2_8x1_flyingchairs_320x448",
    "liteflownet/liteflownet_pre_M5S5R5_8x1_flyingchairs_320x448",
    "liteflownet/liteflownet_pre_M3S3R3_8x1_flyingchairs_320x448",
    "liteflownet/liteflownet_ft_4x1_500k_sintel_384x768",
    "liteflownet2/liteflownet2_pre_M5S5R5_8x1_flyingchairs_320x448",
    "liteflownet2/liteflownet2_pre_M4S4R4_8x1_flyingchairs_320x448",
    "liteflownet2/liteflownet2_8x1_500k_flyingthing3d_subset_384x768",
    "liteflownet2/liteflownet2_pre_M6S6_8x1_flyingchairs_320x448",
    "liteflownet2/liteflownet2_ft_4x1_500k_kitti_320x896",
    "liteflownet2/liteflownet2_pre_M6S6R6_8x1_flyingchairs_320x448",
    "liteflownet2/liteflownet2_ft_4x1_600k_sintel_kitti_320x768",
    "liteflownet2/liteflownet2_pre_M3S3R3_8x1_flyingchairs_320x448",
    "maskflownet/maskflownet_8x1_500k_flyingthings3d_subset_384x768",
    "maskflownet/maskflownets_8x1_sfine_flyingthings3d_subset_384x768",
    "maskflownet/maskflownet_8x1_800k_flyingchairs_384x448",
    "maskflownet/maskflownets_8x1_slong_flyingchairs_384x448",
    "pwcnet/pwcnet_ft_4x1_300k_kitti_320x896",
    "pwcnet/pwcnet_8x1_sfine_flyingthings3d_subset_384x768",
    "pwcnet/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768",
    "pwcnet/pwcnet_ft_4x1_300k_sintel_384x768",
    "pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768",
    "pwcnet/pwcnet_8x1_slong_flyingchairs_384x448",
    "raft/raft_8x2_100k_flyingthings3d_sintel_368x768",
    "raft/raft_8x2_100k_mixed_368x768",  # ****************
    "raft/raft_8x2_100k_flyingchairs_368x496",
    "raft/raft_8x2_100k_flyingthings3d_400x720",
    "raft/raft_8x2_50k_kitti2015_288x960",
]


def get_prediction_fn(model, device="cuda"):
    try:
        from mmflow.apis import inference_model, init_model
    except:
        print()
        print("ERROR: you must install mmflow to use these optical flow models:")
        print("pip install mmflow mmcv-full")
        print()
        exit(1)

    config_file = f"{os.path.dirname(os.path.abspath(__file__))}/../submodules/mmflow/configs/{model}.py"
    checkpoint_file = f"modelzoo/{model.split('/')[-1]}.pth"
    if not os.path.exists(checkpoint_file):
        download(f"https://download.openmmlab.com/mmflow/{model}.pth", checkpoint_file)
    model = init_model(config_file, checkpoint_file, device=device)
    return (
        lambda img1, img2: torch.from_numpy(
            inference_model(
                model,
                img1.detach().squeeze().permute(1, 2, 0).cpu().numpy(),
                img2.detach().squeeze().permute(1, 2, 0).cpu().numpy(),
            )
        )
        .unsqueeze(0)
        .to(img1.device)
    )
