"""Fast differentiable ranking (Blondel et al. 2020), vendored to replace the ``torchsort`` dep.

``torchsort.soft_rank`` computes its isotonic-regression solve in a C++/CUDA kernel that does not
build against this torch/CUDA stack. This absorbs the equivalent pure-Python path: the PAV solvers
come from google-research/fast-soft-sort (third_party/isotonic.py, scikit-learn BSD-3), and the
autograd wrapper mirrors torchsort's ``SoftRank`` (teddykoker/torchsort, Apache-2.0). Only the L2
regularizer is differentiable; the KL forward is provided but its backward is not needed here
(maua only ever calls ``regularization="l2"``).
"""

import numpy as np
import torch

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is an optional speedup
    def njit(func):
        return func


@njit
def _isotonic_l2(y, sol):
    """argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2 via PAV, written into ``sol``."""
    n = y.shape[0]
    target = np.arange(n)
    c = np.ones(n)
    sums = np.zeros(n)
    for i in range(n):
        sol[i] = y[i]
        sums[i] = y[i]
    i = 0
    while i < n:
        k = target[i] + 1
        if k == n:
            break
        if sol[i] > sol[k]:
            i = k
            continue
        sum_y = sums[i]
        sum_c = c[i]
        while True:
            prev_y = sol[k]
            sum_y += sums[k]
            sum_c += c[k]
            k = target[k] + 1
            if k == n or prev_y > sol[k]:
                sol[i] = sum_y / sum_c
                sums[i] = sum_y
                c[i] = sum_c
                target[i] = k - 1
                target[k - 1] = i
                if i > 0:
                    i = target[i - 1]
                break
    i = 0
    while i < n:
        k = target[i] + 1
        sol[i + 1 : k] = sol[i]
        i = k


@njit
def _log_add_exp(x, y):
    larger = max(x, y)
    smaller = min(x, y)
    return larger + np.log1p(np.exp(smaller - larger))


@njit
def _isotonic_kl(y, w, sol):
    """argmin_{v_1 >= ... >= v_n} <e^{y-v}, 1> + <e^w, v> via PAV, written into ``sol``."""
    n = y.shape[0]
    target = np.arange(n)
    lse_y_ = np.zeros(n)
    lse_w_ = np.zeros(n)
    for i in range(n):
        sol[i] = y[i] - w[i]
        lse_y_[i] = y[i]
        lse_w_[i] = w[i]
    i = 0
    while i < n:
        k = target[i] + 1
        if k == n:
            break
        if sol[i] > sol[k]:
            i = k
            continue
        lse_y = lse_y_[i]
        lse_w = lse_w_[i]
        while True:
            prev_y = sol[k]
            lse_y = _log_add_exp(lse_y, lse_y_[k])
            lse_w = _log_add_exp(lse_w, lse_w_[k])
            k = target[k] + 1
            if k == n or prev_y > sol[k]:
                sol[i] = lse_y - lse_w
                lse_y_[i] = lse_y
                lse_w_[i] = lse_w
                target[i] = k - 1
                target[k - 1] = i
                if i > 0:
                    i = target[i - 1]
                break
    i = 0
    while i < n:
        k = target[i] + 1
        sol[i + 1 : k] = sol[i]
        i = k


def _solve_rows(fn, *rows_args):
    """Apply a 1d in-place PAV solver over each row of the batched numpy inputs."""
    first = rows_args[0]
    out = np.empty_like(first)
    for r in range(first.shape[0]):
        sol = np.empty(first.shape[1], dtype=first.dtype)
        fn(*[a[r] for a in rows_args], sol)
        out[r] = sol
    return out


def _isotonic_l2_backward_np(sol, grad):
    """VJP of the L2 isotonic projection: average the incoming grad within each solution block."""
    out = np.empty_like(grad)
    for r in range(sol.shape[0]):
        start = 0
        row_sol, row_grad, row_out = sol[r], grad[r], out[r]
        n = row_sol.shape[0]
        while start < n:
            end = start + 1
            while end < n and abs(row_sol[end] - row_sol[start]) <= 1e-9:
                end += 1
            row_out[start:end] = row_grad[start:end].mean()
            start = end
    return out


def _arange_like(x, reverse=False):
    if reverse:
        ar = torch.arange(x.shape[1] - 1, -1, -1, dtype=x.dtype, device=x.device)
    else:
        ar = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)
    return ar.expand(x.shape[0], -1)


def _inv_permutation(permutation):
    inv = torch.zeros_like(permutation)
    inv.scatter_(1, permutation, _arange_like(permutation).long())
    return inv


class _SoftRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, regularization="l2", regularization_strength=1.0):
        ctx.scale = 1.0 / regularization_strength
        ctx.regularization = regularization
        w = _arange_like(tensor, reverse=True) + 1
        theta = tensor * ctx.scale
        s, permutation = torch.sort(theta, descending=True)
        inv_permutation = _inv_permutation(permutation)

        s_np = s.detach().double().cpu().numpy()
        w_np = w.detach().double().cpu().numpy()
        if regularization == "l2":
            dual_np = _solve_rows(_isotonic_l2, s_np - w_np)
        elif regularization == "kl":
            dual_np = _solve_rows(_isotonic_kl, s_np, np.log(w_np))
        else:
            raise ValueError(f"regularization must be 'l2' or 'kl', got {regularization!r}")
        dual_sol = torch.as_tensor(dual_np, dtype=s.dtype, device=s.device)

        if regularization == "l2":
            ret = (s - dual_sol).gather(1, inv_permutation)
            factor = torch.ones((), dtype=s.dtype, device=s.device)
        else:
            ret = torch.exp((s - dual_sol).gather(1, inv_permutation))
            factor = ret

        ctx.save_for_backward(factor, s, dual_sol, permutation, inv_permutation)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        factor, s, dual_sol, permutation, inv_permutation = ctx.saved_tensors
        grad = (grad_output * factor).clone()
        if ctx.regularization != "l2":
            raise NotImplementedError("backward for the KL regularizer is not vendored (unused by maua)")
        gathered = grad.gather(1, permutation).detach().double().cpu().numpy()
        dual_np = dual_sol.detach().double().cpu().numpy()
        corr = torch.as_tensor(_isotonic_l2_backward_np(dual_np, gathered), dtype=grad.dtype, device=grad.device)
        grad = grad - corr.gather(1, inv_permutation)
        return grad * ctx.scale, None, None


def soft_rank(values, regularization="l2", regularization_strength=1.0):
    """Differentiable soft rank of ``values`` (2d, batch-first), matching ``torchsort.soft_rank``."""
    if values.dim() != 2:
        raise ValueError(f"'values' should be a 2d-tensor but got {values.shape}")
    return _SoftRank.apply(values, regularization, regularization_strength)
