"""Representation-similarity distances vendored from anatome (moskomule/anatome, Apache-2.0).

Only the closed-form distance functions used by ``correlation.py`` are absorbed here so the
module works without the (unpublished-on-PyPI, modern-torch-incompatible) ``anatome`` package.
``_svd`` is reimplemented over ``torch.linalg.svd`` to avoid anatome's ``.utils`` dependency.
"""

from functools import partial

import torch
from torch import Tensor


def _svd(input: Tensor):
    U, S, Vh = torch.linalg.svd(input, full_matrices=False)
    return U, S, Vh.transpose(-2, -1)


def _zero_mean(input: Tensor, dim: int) -> Tensor:
    return input - input.mean(dim=dim, keepdim=True)


def _check_shape_equal(x: Tensor, y: Tensor, dim: int):
    if x.size(dim) != y.size(dim):
        raise ValueError(f"x.size({dim}) == y.size({dim}) is expected, but got {x.size(dim)=}, {y.size(dim)=}.")


def cca_by_svd(x: Tensor, y: Tensor):
    u_1, s_1, v_1 = _svd(x)
    u_2, s_2, v_2 = _svd(y)
    uu = u_1.t() @ u_2
    u, diag, v = _svd(uu)
    a = v_1 @ (1 / s_1[:, None] * u)
    b = v_2 @ (1 / s_2[:, None] * v)
    return a, b, diag


def cca_by_qr(x: Tensor, y: Tensor):
    q_1, r_1 = torch.linalg.qr(x)
    q_2, r_2 = torch.linalg.qr(y)
    qq = q_1.t() @ q_2
    u, diag, v = _svd(qq)
    a = torch.linalg.solve(r_1, u)
    b = torch.linalg.solve(r_2, v)
    return a, b, diag


def cca(x: Tensor, y: Tensor, backend: str):
    """Canonical Correlation Analysis. Returns (x-side coeffs, y-side coeffs, diagonal)."""
    _check_shape_equal(x, y, 0)
    if x.size(0) < x.size(1):
        raise ValueError(f"x.size(0) >= x.size(1) is expected, but got {x.size()=}.")
    if y.size(0) < y.size(1):
        raise ValueError(f"y.size(0) >= y.size(1) is expected, but got {y.size()=}.")
    if backend not in ("svd", "qr"):
        raise ValueError(f"backend is svd or qr, but got {backend}")
    x = _zero_mean(x, dim=0)
    y = _zero_mean(y, dim=0)
    return cca_by_svd(x, y) if backend == "svd" else cca_by_qr(x, y)


def _svd_reduction(input: Tensor, accept_rate: float) -> Tensor:
    left, diag, right = _svd(input)
    full = diag.abs().sum()
    ratio = diag.abs().cumsum(dim=0) / full
    num = torch.where(
        ratio < accept_rate,
        input.new_ones(1, dtype=torch.long),
        input.new_zeros(1, dtype=torch.long),
    ).sum()
    return input @ right[:, :num]


def svcca_distance(x: Tensor, y: Tensor, accept_rate: float, backend: str) -> Tensor:
    """Singular Vector CCA (Raghu et al. 2017)."""
    x = _svd_reduction(x, accept_rate)
    y = _svd_reduction(y, accept_rate)
    div = min(x.size(1), y.size(1))
    a, b, diag = cca(x, y, backend)
    return 1 - diag.sum() / div


def pwcca_distance(x: Tensor, y: Tensor, backend: str) -> Tensor:
    """Projection Weighted CCA (Marcos et al. 2018)."""
    a, b, diag = cca(x, y, backend)
    a, _ = torch.linalg.qr(a)  # reorthonormalize
    alpha = (x @ a).abs_().sum(dim=0)
    alpha /= alpha.sum()
    return 1 - alpha @ diag


def _debiased_dot_product_similarity(z, sum_row_x, sum_row_y, sq_norm_x, sq_norm_y, size):
    return z - size / (size - 2) * (sum_row_x @ sum_row_y) + sq_norm_x * sq_norm_y / ((size - 1) * (size - 2))


def linear_cka_distance(x: Tensor, y: Tensor, reduce_bias: bool) -> Tensor:
    """Linear CKA (Kornblith et al. 2019)."""
    _check_shape_equal(x, y, 0)
    x = _zero_mean(x, dim=0)
    y = _zero_mean(y, dim=0)
    dot_prod = (y.t() @ x).norm("fro").pow(2)
    norm_x = (x.t() @ x).norm("fro")
    norm_y = (y.t() @ y).norm("fro")
    if reduce_bias:
        size = x.size(0)
        sum_row_x = torch.einsum("ij,ij->i", x, x)
        sum_row_y = torch.einsum("ij,ij->i", y, y)
        sq_norm_x = sum_row_x.sum()
        sq_norm_y = sum_row_y.sum()
        dot_prod = _debiased_dot_product_similarity(dot_prod, sum_row_x, sum_row_y, sq_norm_x, sq_norm_y, size)
        norm_x = _debiased_dot_product_similarity(norm_x.pow(2), sum_row_x, sum_row_x, sq_norm_x, sq_norm_x, size).sqrt()
        norm_y = _debiased_dot_product_similarity(norm_y.pow(2), sum_row_y, sum_row_y, sq_norm_y, sq_norm_y, size).sqrt()
    return 1 - dot_prod / (norm_x * norm_y)


def orthogonal_procrustes_distance(x: Tensor, y: Tensor) -> Tensor:
    """Orthogonal Procrustes distance (Ding et al. 2021)."""
    _check_shape_equal(x, y, 0)
    frobenius_norm = partial(torch.linalg.norm, ord="fro")
    nuclear_norm = partial(torch.linalg.norm, ord="nuc")
    x = _zero_mean(x, dim=0)
    x = x / frobenius_norm(x)
    y = _zero_mean(y, dim=0)
    y = y / frobenius_norm(y)
    return 1 - nuclear_norm(x.t() @ y)
