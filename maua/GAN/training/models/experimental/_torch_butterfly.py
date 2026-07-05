"""Real-valued subset of torch_butterfly (HazyResearch/butterfly, Apache-2.0), vendored.

The upstream ``torch_butterfly`` runs ``butterfly_multiply`` through a C++/CUDA op
(``torch.ops.torch_butterfly.*``) that does not build against this torch/CUDA stack, and it pulls
in complex-number and base-4 helpers this codebase never uses. Absorbed here: the pure-Python
``butterfly_multiply_torch`` (aliased as ``butterfly_multiply``) plus the real-only path of the
``Butterfly`` layer used by ``stylehypermixerfly.py``. Complex / fft_no_br / base-4 branches are
dropped.
"""

import math
import numbers

import torch
import torch.nn.functional as F
from torch import nn


def butterfly_multiply(twiddle, input, increasing_stride=True, output_size=None):
    batch_size, nstacks, input_size = input.shape
    nblocks = twiddle.shape[1]
    log_n = twiddle.shape[2]
    n = 1 << log_n
    assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
    input = F.pad(input, (0, n - input_size)) if input_size < n else input[:, :, :n]
    output_size = n if output_size is None else output_size
    assert output_size <= n
    output = input.contiguous()
    cur_increasing_stride = increasing_stride
    for block in range(nblocks):
        for idx in range(log_n):
            log_stride = idx if cur_increasing_stride else log_n - 1 - idx
            stride = 1 << log_stride
            t = twiddle[:, block, idx].view(nstacks, n // (2 * stride), stride, 2, 2).permute(0, 1, 3, 4, 2)
            output_reshape = output.view(batch_size, nstacks, n // (2 * stride), 1, 2, stride)
            output = (t * output_reshape).sum(dim=4)
        cur_increasing_stride = not cur_increasing_stride
    return output.view(batch_size, nstacks, n)[:, :, :output_size]


class Butterfly(nn.Module):
    """Product of log N butterfly factors (real-valued). Compatible with torch.nn.Linear."""

    def __init__(self, in_size, out_size, bias=True, increasing_stride=True, init="randn", nblocks=1):
        super().__init__()
        self.in_size = in_size
        self.log_n = log_n = int(math.ceil(math.log2(in_size)))
        self.n = n = 1 << log_n
        self.out_size = out_size
        self.nstacks = int(math.ceil(out_size / self.n))
        self.complex = False
        self.increasing_stride = increasing_stride
        assert nblocks >= 1
        self.nblocks = nblocks
        dtype = torch.get_default_dtype()
        twiddle_shape = (self.nstacks, nblocks, log_n, n // 2, 2, 2)
        if isinstance(init, torch.Tensor):
            self.init = None
            assert init.shape == twiddle_shape
            assert init.dtype == dtype
            self.twiddle = nn.Parameter(init.clone())
        else:
            assert init in ["empty", "randn", "ortho", "identity"]
            self.init = init
            self.twiddle = nn.Parameter(torch.empty(twiddle_shape, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.twiddle._is_structured = True  # Flag to avoid weight decay
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)
        twiddle = self.twiddle
        if self.init is None or self.init == "empty":
            return
        elif self.init == "randn":
            scaling = 1.0 / math.sqrt(2)
            with torch.no_grad():
                twiddle.copy_(torch.randn(twiddle.shape, dtype=twiddle.dtype) * scaling)
        elif self.init == "ortho":
            twiddle_core_shape = twiddle.shape[:-2]
            theta = torch.rand(twiddle_core_shape) * math.pi * 2
            c, s = torch.cos(theta), torch.sin(theta)
            det = torch.randint(0, 2, twiddle_core_shape, dtype=c.dtype) * 2 - 1  # rotation (+1) or reflection (-1)
            with torch.no_grad():
                twiddle.copy_(
                    torch.stack(
                        (torch.stack((det * c, -det * s), dim=-1), torch.stack((s, c), dim=-1)),
                        dim=-2,
                    )
                )
        elif self.init == "identity":
            twiddle_eye = torch.eye(2, dtype=twiddle.dtype).reshape(1, 1, 1, 1, 2, 2)
            twiddle_eye = twiddle_eye.expand(*twiddle.shape).contiguous()
            with torch.no_grad():
                twiddle.copy_(twiddle_eye)

    def forward(self, input, transpose=False, subtwiddle=False):
        twiddle = self.twiddle
        output = self.pre_process(input)
        output_size = self.out_size if self.nstacks == 1 else None
        if subtwiddle:
            log_n = int(math.ceil(math.log2(input.size(-1))))
            n = 1 << log_n
            twiddle = twiddle[:, :, :log_n, : n // 2] if self.increasing_stride else twiddle[:, :, -log_n:, : n // 2]
            output_size = None
        if not transpose:
            output = butterfly_multiply(twiddle, output, self.increasing_stride, output_size)
        else:
            twiddle = twiddle.transpose(-1, -2).flip([1, 2])
            last_increasing_stride = self.increasing_stride != ((self.nblocks - 1) % 2 == 1)
            output = butterfly_multiply(twiddle, output, not last_increasing_stride, output_size)
        if not subtwiddle:
            return self.post_process(input, output)
        return self.post_process(input, output, out_size=output.size(-1))

    def pre_process(self, input):
        input_size = input.size(-1)
        output = input.reshape(-1, input_size)
        batch = output.shape[0]
        output = output.unsqueeze(1).expand(batch, self.nstacks, input_size)
        return output

    def post_process(self, input, output, out_size=None):
        if out_size is None:
            out_size = self.out_size
        batch = output.shape[0]
        output = output.view(batch, self.nstacks * output.size(-1))
        if out_size != output.shape[-1]:  # take top rows
            output = output[:, :out_size]
        if self.bias is not None:
            output = output + self.bias[:out_size]
        return output.view(*input.size()[:-1], out_size)

    def __imul__(self, scale):
        assert isinstance(scale, numbers.Number)
        assert scale >= 0
        self.twiddle *= scale ** (1.0 / self.twiddle.shape[1] / self.twiddle.shape[2])
        return self

    def extra_repr(self):
        return (
            f"in_size={self.in_size}, out_size={self.out_size}, bias={self.bias is not None}, "
            f"increasing_stride={self.increasing_stride}, init={self.init}, nblocks={self.nblocks}"
        )
