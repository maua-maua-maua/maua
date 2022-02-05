import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor


# fmt:off
activation_funcs = {
    'linear':   dict(func=lambda x, **_:        x,                                        def_alpha=0.,  def_gain=1.,           has_2nd_grad=False),
    'relu':     dict(func=lambda x, **_:        torch.nn.functional.relu(x),              def_alpha=0.,  def_gain=math.sqrt(2), has_2nd_grad=False),
    'lrelu':    dict(func=lambda x, alpha, **_: torch.nn.functional.leaky_relu(x, alpha), def_alpha=0.2, def_gain=math.sqrt(2), has_2nd_grad=False),
    'tanh':     dict(func=lambda x, **_:        torch.tanh(x),                            def_alpha=0.,  def_gain=1.,           has_2nd_grad=True),
    'sigmoid':  dict(func=lambda x, **_:        torch.sigmoid(x),                         def_alpha=0.,  def_gain=1.,           has_2nd_grad=True),
    'elu':      dict(func=lambda x, **_:        torch.nn.functional.elu(x),               def_alpha=0.,  def_gain=1.,           has_2nd_grad=True),
    'selu':     dict(func=lambda x, **_:        torch.nn.functional.selu(x),              def_alpha=0.,  def_gain=1.,           has_2nd_grad=True),
    'softplus': dict(func=lambda x, **_:        torch.nn.functional.softplus(x),          def_alpha=0.,  def_gain=1.,           has_2nd_grad=True),
    'swish':    dict(func=lambda x, **_:        torch.sigmoid(x) * x,                     def_alpha=0.,  def_gain=math.sqrt(2), has_2nd_grad=True),
}
# fmt:on


def get_activation_defaults(activation: str) -> Tuple[float, float]:
    if activation == "relu":
        return 0.0, math.sqrt(2)
    elif activation == "lrelu":
        return 0.2, math.sqrt(2)
    elif activation == "tanh":
        return 0.0, 1.0
    elif activation == "sigmoid":
        return 0.0, 1.0
    elif activation == "elu":
        return 0.0, 1.0
    elif activation == "selu":
        return 0.0, 1.0
    elif activation == "softplus":
        return 0.0, 1.0
    elif activation == "swish":
        return 0.0, math.sqrt(2)
    else:
        return 0.0, 1.0


def activate(x: Tensor, act: str, alpha: float):
    if act == "relu":
        return torch.nn.functional.relu(x)
    elif act == "lrelu":
        return torch.nn.functional.leaky_relu(x, alpha)
    elif act == "tanh":
        return torch.tanh(x)
    elif act == "sigmoid":
        return torch.sigmoid(x)
    elif act == "elu":
        return torch.nn.functional.elu(x)
    elif act == "selu":
        return torch.nn.functional.selu(x)
    elif act == "softplus":
        return torch.nn.functional.softplus(x)
    elif act == "swish":
        return torch.sigmoid(x) * x
    else:
        return x


def bias_act(
    x: Tensor,
    b: Optional[Tensor] = None,
    act: str = "linear",
    alpha: Optional[float] = None,
    gain: Optional[float] = None,
    clamp: Optional[float] = None,
):
    def_alpha, def_gain = get_activation_defaults(act)
    alpha = float(alpha if alpha is not None else def_alpha)
    gain = float(gain if gain is not None else def_gain)
    clamp = float(clamp if clamp is not None else -1.0)
    if b is not None:
        x = x + b.reshape([-1 if i == 1 else 1 for i in range(x.ndim)])
    x = activate(x, act, alpha)
    if gain != 1:
        x = x * gain
    if clamp >= 0:
        x = x.clamp(-clamp, clamp)  # pylint: disable=invalid-unary-operand-type
    return x


def repeat2(scaling: int) -> Tuple[int, int]:
    return scaling, scaling


def repeat4(padding: int) -> Tuple[int, int, int, int]:
    return padding, padding, padding, padding


def zero_pad(x, padding):
    if sum(padding) == 0:
        return x

    if len(padding) == 4:
        b, bb, one, two = x.shape
        one1, one2, two1, two2 = padding

        tensors = []
        if one1 > 0:
            x1 = torch.zeros((b, bb, one1, two), dtype=x.dtype, device=x.device)
            tensors.append(x1)
        tensors.append(x)
        if one2 > 0:
            x2 = torch.zeros((b, bb, one2, two), dtype=x.dtype, device=x.device)
            tensors.append(x2)
        x = torch.cat(tensors, dim=2)

        tensors = []
        if two1 > 0:
            x3 = torch.zeros((b, bb, one + one1 + one2, two1), dtype=x.dtype, device=x.device)
            tensors.append(x3)
        tensors.append(x)
        if two2 > 0:
            x4 = torch.zeros((b, bb, one + one1 + one2, two2), dtype=x.dtype, device=x.device)
            tensors.append(x4)
        x = torch.cat(tensors, dim=3)

    if len(padding) == 6:
        b, bb, bbb, one, two, thr = x.shape
        one1, one2, two1, two2, thr1, thr2 = padding

        tensors = []
        if one1 > 0:
            x1 = torch.zeros((b, bb, bbb, one1, two, thr), dtype=x.dtype, device=x.device)
            tensors.append(x1)
        tensors.append(x)
        if one2 > 0:
            x2 = torch.zeros((b, bb, bbb, one2, two, thr), dtype=x.dtype, device=x.device)
            tensors.append(x2)
        x = torch.cat(tensors, dim=3)

        tensors = []
        if two1 > 0:
            x3 = torch.zeros((b, bb, bbb, one + one1 + one2, two1, thr), dtype=x.dtype, device=x.device)
            tensors.append(x3)
        tensors.append(x)
        if two2 > 0:
            x4 = torch.zeros((b, bb, bbb, one + one1 + one2, two2, thr), dtype=x.dtype, device=x.device)
            tensors.append(x4)
        x = torch.cat(tensors, dim=4)

        tensors = []
        if thr1 > 0:
            x5 = torch.zeros((b, bb, bbb, one + one1 + one2, two + two1 + two2, thr1), dtype=x.dtype, device=x.device)
            tensors.append(x5)
        tensors.append(x)
        if thr2 > 0:
            x6 = torch.zeros((b, bb, bbb, one + one1 + one2, two + two1 + two2, thr2), dtype=x.dtype, device=x.device)
            tensors.append(x6)
        x = torch.cat(tensors, dim=5)

    return x


def upfirdn2d(
    x: Tensor,
    f: Optional[Tensor],
    up: int = 1,
    down: int = 1,
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    gain: float = 1,
):
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = repeat2(up)
    downx, downy = repeat2(down)
    padx0, padx1, pady0, pady1 = padding
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = zero_pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])
    x = zero_pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]
    f = f * (gain ** (f.ndim / 2))
    f = f.unsqueeze(0).unsqueeze(0)
    f = f * torch.tensor([1] * num_channels, dtype=f.dtype, device=f.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    if f.ndim == 4:
        x = torch.nn.functional.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = torch.nn.functional.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = torch.nn.functional.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)
    x = x[:, :, ::downy, ::downx]
    return x


def upsample2d(x: Tensor, f: Tensor, up: int = 2, padding: int = 0, gain: float = 1):
    upx, upy = repeat2(up)
    padx0, padx1, pady0, pady1 = repeat4(padding)
    fw, fh = _get_filter_size(f)
    p = (
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    )
    return upfirdn2d(x, f, up=up, padding=p, gain=gain * upx * upy)


def _get_filter_size(f: Optional[Tensor]) -> Tuple[int, int]:
    if f is None:
        return 1, 1
    return int(f.shape[-1]), int(f.shape[0])


def conv2d_resample(
    x: Tensor,
    w: Tensor,
    f: Optional[Tensor] = None,
    up: int = 1,
    down: int = 1,
    padding: int = 0,
    groups: int = 1,
):
    out_channels, in_channels_per_group, kh, kw = [int(sz) for sz in w.shape]
    fw, fh = _get_filter_size(f)
    px0, px1, py0, py1 = repeat4(padding)
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2
    if up > 1:
        if groups == 1:
            w = w.permute(1, 0, 2, 3, 4)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kh, kw)
            w = w.permute(0, 2, 1, 3, 4)
            w = w.reshape(groups * in_channels_per_group, out_channels // groups, kh, kw)
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        x = torch.nn.functional.conv_transpose2d(x, w, stride=up, padding=(pyt, pxt), groups=groups)
        x = upfirdn2d(x=x, f=f, padding=(px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt), gain=up ** 2)
        if down > 1:
            x = upfirdn2d(x=x, f=f, down=down)
        return x
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return torch.nn.functional.conv2d(x, w, padding=(py0, px0), groups=groups)
    assert False, "Something weird is going on..."
    return x  # THIS PATH IS NEVER TAKEN!


def setup_filter(
    f: List[int],
    device: torch.device = torch.device("cpu"),
    normalize: bool = True,
    gain: float = 1,
    separable: Optional[bool] = None,
):
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    if f.ndim == 0:
        f = f[None]
    if separable is None:
        separable = f.ndim == 1 and f.numel() >= 8
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    if normalize:
        f /= f.sum()
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f
