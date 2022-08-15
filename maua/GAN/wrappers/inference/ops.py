from math import sqrt
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import pad

# fmt:off
activation_funcs = {
    'linear':   dict(func=lambda x, **_:        x,                                        def_alpha=torch.tensor(0.) , def_gain=torch.tensor(1.),           has_2nd_grad=False),
    'relu':     dict(func=lambda x, **_:        torch.nn.functional.relu(x),              def_alpha=torch.tensor(0.) , def_gain=torch.tensor(sqrt(2)), has_2nd_grad=False),
    'lrelu':    dict(func=lambda x, alpha, **_: torch.nn.functional.leaky_relu(x, alpha), def_alpha=torch.tensor(0.2), def_gain=torch.tensor(sqrt(2)), has_2nd_grad=False),
    'tanh':     dict(func=lambda x, **_:        torch.tanh(x),                            def_alpha=torch.tensor(0.) , def_gain=torch.tensor(1.),           has_2nd_grad=True),
    'sigmoid':  dict(func=lambda x, **_:        torch.sigmoid(x),                         def_alpha=torch.tensor(0.) , def_gain=torch.tensor(1.),           has_2nd_grad=True),
    'elu':      dict(func=lambda x, **_:        torch.nn.functional.elu(x),               def_alpha=torch.tensor(0.) , def_gain=torch.tensor(1.),           has_2nd_grad=True),
    'selu':     dict(func=lambda x, **_:        torch.nn.functional.selu(x),              def_alpha=torch.tensor(0.) , def_gain=torch.tensor(1.),           has_2nd_grad=True),
    'softplus': dict(func=lambda x, **_:        torch.nn.functional.softplus(x),          def_alpha=torch.tensor(0.) , def_gain=torch.tensor(1.),           has_2nd_grad=True),
    'swish':    dict(func=lambda x, **_:        torch.sigmoid(x) * x,                     def_alpha=torch.tensor(0.) , def_gain=torch.tensor(sqrt(2)), has_2nd_grad=True),
}
# fmt:on


def get_activation_defaults(activation: str):
    if activation == "relu":
        return torch.tensor(0.0), torch.tensor(sqrt(2))
    elif activation == "lrelu":
        return torch.tensor(0.2), torch.tensor(sqrt(2))
    elif activation == "tanh":
        return torch.tensor(0.0), torch.tensor(1.0)
    elif activation == "sigmoid":
        return torch.tensor(0.0), torch.tensor(1.0)
    elif activation == "elu":
        return torch.tensor(0.0), torch.tensor(1.0)
    elif activation == "selu":
        return torch.tensor(0.0), torch.tensor(1.0)
    elif activation == "softplus":
        return torch.tensor(0.0), torch.tensor(1.0)
    elif activation == "swish":
        return torch.tensor(0.0), torch.tensor(sqrt(2))
    else:
        return torch.tensor(0.0), torch.tensor(1.0)


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
    alpha: Optional[Tensor] = None,
    gain: Optional[Tensor] = None,
    clamp: Optional[Tensor] = None,
):
    def_alpha, def_gain = get_activation_defaults(act)
    alpha = alpha if alpha is not None else def_alpha
    gain = gain if gain is not None else def_gain
    clamp = clamp if clamp is not None else torch.tensor(-1.0)
    if b is not None:
        x = x + b.reshape([-1 if i == 1 else 1 for i in range(x.ndim)])
    x = activate(x, act, alpha)
    if gain != 1:
        x = x * gain
    if clamp >= 0:
        x = x.clamp(-clamp, clamp)  # pylint: disable=invalid-unary-operand-type
    return x


def upfirdn2d(
    x: Tensor,
    f: Optional[Tensor],
    up: Tensor = torch.tensor(1),
    down: Tensor = torch.tensor(1),
    padding: Tensor = torch.zeros(4),
    gain: Tensor = torch.tensor(1),
):
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = up.repeat(2)
    downx, downy = down.repeat(2)
    padx0, padx1, pady0, pady1 = padding
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])
    x = pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]
    f = f * (gain ** (f.ndim / 2))
    f = f[None, None].repeat(num_channels, 1, 1, 1)
    if f.ndim == 4:
        x = torch.nn.functional.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = torch.nn.functional.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = torch.nn.functional.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)
    x = x[:, :, ::downy, ::downx]
    return x


def upsample2d(
    x: Tensor,
    f: Tensor,
    up: Tensor = torch.tensor(2),
    padding: Tensor = torch.tensor(0),
    gain: Tensor = torch.tensor(1),
):
    upx, upy = up.repeat(2)
    padx0, padx1, pady0, pady1 = padding.repeat(4)
    fw, fh = _get_filter_size(f)
    p = (
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    )
    return upfirdn2d(x, f, up=up, padding=p, gain=gain * upx * upy)


def _get_filter_size(f: Optional[Tensor]):
    if f is None:
        return torch.ones((2))
    return f.shape[-1], f.shape[0]


def normalize_2nd_moment(x, dim: Tensor = 1, eps: float = 1e-8):
    return x / ((x * x).mean(dim=dim, keepdim=True) + eps).sqrt()


def modulated_conv2d(
    x: Tensor,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight: Tensor,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles: Tensor,  # Modulation coefficients of shape [batch_size, in_channels].
    noise: Optional[Tensor] = None,  # Optional noise tensor to add to the output activations.
    up: Tensor = torch.tensor(1),  # Integer upsampling factor.
    down: Tensor = torch.tensor(1),  # Integer downsampling factor.
    padding: Tensor = torch.tensor(0),  # Padding with respect to the upsampled image.
    resample_filter: Optional[Tensor] = None,  # Low-pass filter to apply when resampling activations.
    demodulate: bool = True,  # Apply weight demodulation?
):
    B, xc, xh, xw = x.shape
    wco, wci, kh, kw = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight / (
            torch.amax(torch.abs(weight), dim=(1, 2, 3)).reshape(weight.shape[0], 1, 1, 1) * sqrt(wci * kh * kw)
        )
        styles = styles / torch.max(torch.abs(styles), dim=1).values.unsqueeze(1)

    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0) * styles.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if demodulate:
        denom = ((w * w).sum((2, 3, 4)) + 1e-8).sqrt()
        w = w / denom.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # Execute as one fused op using grouped convolution.
    x = conv2d_resample(
        x=x.reshape(1, B * xc, xh, xw),
        w=w.reshape(B * wco, wci, kh, kw),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        groups=B,
    )
    x = x.reshape(B, wco, xh * up, xw * up)
    if noise is not None:
        x = x + noise
    return x


def conv2d_resample(
    x: Tensor,
    w: Tensor,
    f: Optional[Tensor] = None,
    up: Tensor = torch.tensor(1),
    down: Tensor = torch.tensor(1),
    padding: Tensor = torch.tensor(0),
    groups: Tensor = torch.tensor(1),
):
    out_channels, in_channels_per_group, kh, kw = w.shape
    fw, fh = _get_filter_size(f)
    px0, px1, py0, py1 = padding.repeat(4)
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
        pxt = torch.max(torch.min(-px0, -px1), 0)
        pyt = torch.max(torch.min(-py0, -py1), 0)
        x = torch.nn.functional.conv_transpose2d(x, w, stride=up, padding=(pyt, pxt), groups=groups)
        x = upfirdn2d(x=x, f=f, padding=(px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt), gain=up**2)
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
    gain: Tensor = torch.tensor(1),
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
