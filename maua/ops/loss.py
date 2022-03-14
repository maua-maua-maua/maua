from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss


class NormalizeGradients(torch.autograd.Function):
    @staticmethod
    def forward(self, input_tensor):
        return input_tensor

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input / (torch.norm(grad_input, keepdim=True) + torch.finfo(grad_input.dtype).eps)
        return grad_input, None


normalize_gradients = NormalizeGradients.apply


def normalize_weights(tensor, strategy: str):
    if strategy == "elements":
        return tensor.numel()
    elif strategy == "channels":
        return tensor.size(1)
    elif strategy == "area":
        return tensor.size(2) * tensor.size(3)
    return 1


def scaled_mse_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """Computes MSE scaled such that its gradient L1 norm is approximately 1."""
    diff = input - target
    return diff.pow(2).sum() / diff.abs().sum().add(eps)


def feature_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    norm_weights: Optional[str] = "elements",
    scaled: bool = True,
):
    if scaled:
        loss = scaled_mse_loss(input, target)
        loss /= normalize_weights(input, norm_weights)
    else:
        loss = mse_loss(input, target)
        loss /= normalize_weights(input, norm_weights)
        loss = normalize_gradients(loss)
    return loss


def gram_matrix(x, shift_x=0, shift_y=0, shift_t=0, flip_h=False, flip_v=False, use_covariance=False):
    B, C, H, W = x.size()

    # maybe apply transforms before calculating gram matrix
    if not (shift_x == 0 and shift_y == 0):
        x = x[:, :, shift_y:, shift_x:]
        y = x[:, :, : H - shift_y, : W - shift_x]
        B, C, H, W = x.size()
    if flip_h:
        y = x[:, :, :, ::-1]
    if flip_v:
        y = x[:, :, ::-1, :]
    else:
        # TODO does this double the required memory?
        y = x

    x_flat = x.reshape(B * C, H * W)
    y_flat = y.reshape(B * C, H * W)

    if use_covariance:
        x_flat = x_flat - x_flat.mean(1).unsqueeze(1)
        y_flat = y_flat - y_flat.mean(1).unsqueeze(1)

    return x_flat @ y_flat.T


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3]).squeeze()


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        (input,) = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def clamp_grad(input, min, max):
    return replace_grad(input.clamp(min, max), input)
