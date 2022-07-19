import numpy as np
import scipy
import torch
from torch.nn.functional import conv2d, grid_sample
from torchvision.transforms.functional import gaussian_blur


def check_consistency_np(flow1, flow2, edges_unreliable=True):
    # algorithm based on https://github.com/manuelruder/artistic-videos/blob/master/consistencyChecker/consistencyChecker.cpp
    # reimplemented in numpy by Hans Brouwer
    # // consistencyChecker
    # // Check consistency of forward flow via backward flow.
    # // (c) Manuel Ruder, Alexey Dosovitskiy, Thomas Brox 2016

    flow1 = np.flip(flow1, axis=2)
    flow2 = np.flip(flow2, axis=2)
    h, w, _ = flow1.shape

    # get grid of coordinates for each pixel
    orig_coord = np.flip(np.mgrid[:w, :h], 0).T

    # find where the flow1 maps each pixel
    warp_coord = orig_coord + flow1

    # clip the coordinates in bounds and round down
    warp_coord_inbound = np.zeros_like(warp_coord)
    warp_coord_inbound[..., 0] = np.clip(warp_coord[..., 0], 0, h - 2)
    warp_coord_inbound[..., 1] = np.clip(warp_coord[..., 1], 0, w - 2)
    warp_coord_floor = np.floor(warp_coord_inbound).astype(np.int)

    # for each pixel: bilinear interpolation of the corresponding flow2 values around the point mapped to by flow1
    alpha = warp_coord_inbound - warp_coord_floor
    flow2_00 = flow2[warp_coord_floor[..., 0], warp_coord_floor[..., 1]]
    flow2_01 = flow2[warp_coord_floor[..., 0], warp_coord_floor[..., 1] + 1]
    flow2_10 = flow2[warp_coord_floor[..., 0] + 1, warp_coord_floor[..., 1]]
    flow2_11 = flow2[warp_coord_floor[..., 0] + 1, warp_coord_floor[..., 1] + 1]
    flow2_0_blend = (1 - alpha[..., 1, None]) * flow2_00 + alpha[..., 1, None] * flow2_01
    flow2_1_blend = (1 - alpha[..., 1, None]) * flow2_10 + alpha[..., 1, None] * flow2_11
    warp_coord_flow2 = (1 - alpha[..., 0, None]) * flow2_0_blend + alpha[..., 0, None] * flow2_1_blend

    # coordinates that flow2 remaps each flow1-mapped pixel to
    rewarp_coord = warp_coord + warp_coord_flow2

    # where the difference in position after flow1 and flow2 are applied is larger than a threshold there is likely an
    # occlusion. set values to -1 so the final gaussian blur will spread the value a couple pixels around this area
    squared_diff = np.sum((rewarp_coord - orig_coord) ** 2, axis=2)
    threshold = 0.01 * np.sum(warp_coord_flow2**2 + flow1**2, axis=2) + 0.5
    reliable_flow = np.where(squared_diff >= threshold, -0.75, 1)

    # areas mapping outside of the frame are also occluded (don't need extra region around these though, so set 0)
    if edges_unreliable:
        reliable_flow = np.where(
            np.logical_or.reduce(
                (
                    warp_coord[..., 0] < 0,
                    warp_coord[..., 1] < 0,
                    warp_coord[..., 0] >= h - 1,
                    warp_coord[..., 1] >= w - 1,
                )
            ),
            0,
            reliable_flow,
        )

    # get derivative of flow, large changes in derivative => edge of moving object
    dx = np.diff(flow1, axis=1, append=0)
    dy = np.diff(flow1, axis=0, append=0)
    motion_edge = np.sum(dx**2 + dy**2, axis=2)
    motion_threshold = 0.01 * np.sum(flow1**2, axis=2) + 0.002
    reliable_flow = np.where(np.logical_and(motion_edge > motion_threshold, reliable_flow != -0.75), 0, reliable_flow)

    # blur and clip values between 0 and 1
    reliable_flow = scipy.ndimage.gaussian_filter(reliable_flow, [3, 3])
    reliable_flow = reliable_flow.clip(0, 1)
    return reliable_flow


def sample(tensor, uv):
    height, width = tensor.shape[-2:]
    max_pos = torch.tensor([width - 1, height - 1], device=tensor.device).view(2, 1, 1)
    grid = uv.div(max_pos / 2).sub(1).movedim(0, -1).unsqueeze(0)
    return grid_sample(tensor.unsqueeze(0), grid, align_corners=True).squeeze(0)


@torch.no_grad()
def check_consistency(flow_forward, flow_backward):
    # algorithm based on https://github.com/manuelruder/artistic-videos/blob/master/consistencyChecker/consistencyChecker.cpp
    # reimplemented in pytorch by Henry Rachootin
    # (c) Manuel Ruder, Alexey Dosovitskiy, Thomas Brox 2016
    dev = flow_forward.device
    batch, height, width, two = flow_forward.shape
    flow_forward, flow_backward = flow_forward.permute(0, 3, 1, 2), flow_backward.permute(0, 3, 1, 2)

    dx_ker = torch.tensor([[[[0, 0, 0], [1, 0, -1], [0, 0, 0]]]], device=dev).float().div(2).repeat(2, 2, 1, 1)
    dy_ker = torch.tensor([[[[0, 1, 0], [0, 0, 0], [0, -1, 0]]]], device=dev).float().div(2).repeat(2, 2, 1, 1)
    f_x = conv2d(flow_backward, dx_ker, padding="same")
    f_y = conv2d(flow_backward, dy_ker, padding="same")
    motionedge = torch.cat([f_x, f_y]).square().sum(dim=(0, 1))

    y, x = torch.meshgrid([torch.arange(0, height, device=dev), torch.arange(0, width, device=dev)], indexing="ij")

    p1 = torch.stack([x, y])
    v1 = flow_forward.squeeze(0)
    p0 = p1 + flow_backward.squeeze()
    v0 = sample(v1, p0)
    p1_back = p0 + v0
    v1_back = flow_backward.squeeze(0)

    r1 = torch.floor(p0)
    r2 = r1 + 1
    max_pos = torch.tensor([width - 1, height - 1], device=dev).view(2, 1, 1)
    min_pos = torch.tensor([0, 0], device=dev).view(2, 1, 1)
    overshoot = torch.logical_or(r1.lt(min_pos), r2.gt(max_pos))
    overshoot = torch.logical_or(overshoot[0], overshoot[1])

    missed = (
        (p1_back - p1).square().sum(dim=0).ge(torch.stack([v1_back, v0]).square().sum(dim=(0, 1)).mul(0.01).add(0.5))
    )
    motion_boundary = motionedge.ge(v1_back.square().sum(dim=0).mul(0.01).add(0.002))

    reliable = torch.ones((height, width), device=dev)
    reliable[motion_boundary] = 0
    reliable[missed] = -0.75
    reliable[overshoot] = 0
    mask = gaussian_blur(reliable.unsqueeze(0), 3).clip(0, 1)

    return mask
