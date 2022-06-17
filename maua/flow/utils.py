"""
Copyright (c) 2019 LI RUOTENG
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch


def resample_flow(img, sz):
    """
    img: flow map to be resampled
    sz: new flow map size. Must be [height,weight]
    """
    in_height = img.shape[0]
    in_width = img.shape[1]
    out_height = sz[0]
    out_width = sz[1]
    out_flow = torch.zeros((out_height, out_width, 2)).to(img)

    height_scale = float(in_height) / float(out_height)
    width_scale = float(in_width) / float(out_width)

    x, y = torch.meshgrid(torch.arange(out_width).to(img), torch.arange(out_height).to(img), indexing="xy")
    xx = x * width_scale
    yy = y * height_scale
    x0 = torch.floor(xx).long()
    x1 = x0 + 1
    y0 = torch.floor(yy).long()
    y1 = y0 + 1

    x0 = torch.clip(x0, 0, in_width - 1)
    x1 = torch.clip(x1, 0, in_width - 1)
    y0 = torch.clip(y0, 0, in_height - 1)
    y1 = torch.clip(y1, 0, in_height - 1)

    Ia = img[y0, x0, :]
    Ib = img[y1, x0, :]
    Ic = img[y0, x1, :]
    Id = img[y1, x1, :]

    wa = (y1 - yy) * (x1 - xx)
    wb = (yy - y0) * (x1 - xx)
    wc = (y1 - yy) * (xx - x0)
    wd = (yy - y0) * (xx - x0)
    out_flow[:, :, 0] = (
        (Ia[:, :, 0] * wa + Ib[:, :, 0] * wb + Ic[:, :, 0] * wc + Id[:, :, 0] * wd) * out_width / in_width
    )
    out_flow[:, :, 1] = (
        (Ia[:, :, 1] * wa + Ib[:, :, 1] * wb + Ic[:, :, 1] * wc + Id[:, :, 1] * wd) * out_height / in_height
    )

    return out_flow


def read_flow(filename):
    with open(filename, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            raise Exception("Magic number incorrect. Invalid .flo file")
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])
        flow = np.fromfile(f, np.float32, count=2 * w * h)
    flow = np.resize(flow, (h, w, 2))
    return flow


def write_flow(flow, filename):
    with open(filename, "wb") as f:
        magic = np.array([202021.25], dtype=np.float32)
        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        magic.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    UNKNOWN_FLOW_THRESH = 1e7

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.0
    maxv = -999.0
    minu = 999.0
    minv = 999.0

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col : col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col : col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += +BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col : col + MR, 0] = 255

    return colorwheel
