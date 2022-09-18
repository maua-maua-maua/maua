import math
from typing import List, Optional

import numpy as np
import torch
from anatome.distance import cca as canonical_correlation_analysis
from anatome.distance import linear_cka_distance, orthogonal_procrustes_distance, pwcca_distance, svcca_distance
from torch import Tensor
from torchmetrics.functional import matthews_corrcoef
from torchsort import soft_rank


@torch.jit.script
def _pearson_correlation(X, Y, batch_first: bool = False):
    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_X = X - X.mean(dim=dim, keepdim=True)
    centered_Y = Y - Y.mean(dim=dim, keepdim=True)

    covariance = (centered_X * centered_Y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (X.shape[dim] - 1)

    X_std = X.std(dim=dim, keepdim=True)
    Y_std = Y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (X_std * Y_std)

    return corr


@torch.jit.script
def _concordance_correlation(X, Y, batch_first: bool = False):
    if batch_first:
        dim = -1
    else:
        dim = 0

    bessel_correction_term = (X.shape[dim] - 1) / X.shape[dim]

    r = _pearson_correlation(X, Y, batch_first)
    X_mean = X.mean(dim=dim, keepdim=True)
    Y_mean = Y.mean(dim=dim, keepdim=True)
    X_std = X.std(dim=dim, keepdim=True)
    Y_std = Y.std(dim=dim, keepdim=True)
    concordance = (
        2
        * r
        * X_std
        * Y_std
        / (X_std * X_std + Y_std * Y_std + (X_mean - Y_mean) * (X_mean - Y_mean) / bessel_correction_term)
    )
    return concordance


def _spearman_correlation(X, Y, regularization: str = "l2", regularization_strength: float = 0.1):
    X = soft_rank(X, regularization, regularization_strength) / X.shape[-1]
    Y = soft_rank(Y, regularization, regularization_strength) / Y.shape[-1]
    return _pearson_correlation(X, Y)


def _matthews_correlation(X, Y, regularization: str = "l2", regularization_strength: float = 0.1):
    X = soft_rank(X, regularization, regularization_strength) / X.shape[-1]
    Y = soft_rank(Y, regularization, regularization_strength) / Y.shape[-1]
    return matthews_corrcoef(X, Y, num_classes=X.shape[1])


@torch.jit.script
def _autocorrelation_correlation(X, Y, center: bool = True):
    if center:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)

    X = X / X.norm(p=2, dim=1, keepdim=True)
    XX = X @ X.T

    Y = Y / Y.norm(p=2, dim=1, keepdim=True)
    YY = Y @ Y.T

    T = len(X)
    triuy, triux = torch.triu_indices(T, T, offset=1).unbind(0)
    similarity = _pearson_correlation(XX[triuy, triux], YY[triuy, triux])
    return similarity


@torch.jit.script
def _rv(Ms: List[Tensor], center: bool = True, modified: bool = True, standardize: bool = False):
    """
    This function computes the RV matrix correlation coefficients between pairs of arrays. The number and order of
    objects (rows) for the two arrays must match. The number of variables in each array may vary. The RV2 coefficient is
    a modified version of the RV coefficient with values -1 <= RV2 <= 1. RV2 is independent of object and variable size.

    Reference: `Matrix correlations for high-dimensional data - the modified RV-coefficient`_
    .. _Matrix correlations for high-dimensional data - the modified RV-coefficient: https://academic.oup.com/bioinformatics/article/25/3/401/244239
    .. _Hoggorm implementation: https://github.com/olivertomic/hoggorm
    """

    Mss = []
    for M in Ms:
        if center:
            M = M - M.mean(0)
        if standardize:
            M = M / M.std()
        MMt = M @ M.T
        if modified:
            MMt = MMt - torch.diag(torch.diag(MMt))
        Mss.append(MMt)

    C = torch.eye(len(Ms), dtype=Ms[0].dtype, device=Ms[0].device)
    for idx in torch.triu_indices(len(Ms), len(Ms), offset=1).T:
        Rv = torch.trace(Mss[idx[0]].T @ Mss[idx[1]]) / torch.sqrt(
            torch.trace(Mss[idx[0]].T @ Mss[idx[0]]) * torch.trace(Mss[idx[1]].T @ Mss[idx[1]])
        )
        C[idx[0], idx[1]] = C[idx[1], idx[0]] = Rv

    if len(Ms) == 2:
        return C[1, 0]
    return C


@torch.jit.script
def _rvadj_maye(X, Y, center: bool = True):
    if center:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)

    n = X.shape[0]
    p = X.shape[1]
    q = Y.shape[1]
    pq = p * q
    pp = p * p
    qq = q * q

    XX = X.T @ X
    YY = Y.T @ Y
    sx = X.std(0)
    sy = Y.std(0)
    msxy = torch.stack([sx.min(), sx.max(), sy.min(), sy.max()])

    if torch.any(msxy > 1 + 1e-12) or torch.any(msxy < 1 - 1e-12):  # Not standardized X/Y
        Xs = X / sx
        Ys = Y / sy
        XXs = Xs.T @ Xs
        YYs = Ys.T @ Ys

        # Find scaling between R2 and R2adj
        xy = torch.trace(XXs @ YYs) / (pq - (n - 1) / (n - 2) * (pq - torch.trace(XXs @ YYs) / (n - 1) ** 2))
        xx = torch.trace(XXs @ XXs) / (pp - (n - 1) / (n - 2) * (pp - torch.trace(XXs @ XXs) / (n - 1) ** 2))
        yy = torch.trace(YYs @ YYs) / (qq - (n - 1) / (n - 2) * (qq - torch.trace(YYs @ YYs) / (n - 1) ** 2))

        # Apply scaling to non-standarized data
        RVadj = (torch.trace(XX @ YY) / xy) / torch.sqrt(torch.trace(XX @ XX) / xx * torch.trace(YY @ YY) / yy)
    else:
        RVadj = (pq - (n - 1) / (n - 2) * (pq - torch.trace(XX @ YY) / (n - 1) ** 2)) / torch.sqrt(
            (pp - (n - 1) / (n - 2) * (pp - torch.trace(XX @ XX) / (n - 1) ** 2))
            * (qq - (n - 1) / (n - 2) * (qq - torch.trace(YY @ YY) / (n - 1) ** 2))
        )
    return RVadj


@torch.jit.script
def _rvadj_ghaziri(X, Y, center: bool = True):
    if center:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
    n = X.shape[0]
    XX = X.T @ X
    YY = Y.T @ Y
    rv = torch.trace(XX @ YY) / (XX @ XX).norm() / (YY @ YY).norm()
    mrvB = (
        torch.sqrt(torch.trace(XX) ** 2 / torch.trace(XX @ XX))
        * torch.sqrt(torch.trace(YY) ** 2 / torch.trace(YY @ YY))
        / (n - 1)
    )
    aRV = (rv - mrvB) / (1 - mrvB)
    return aRV


@torch.jit.script
def _matrix_rank(X, tol: float = 1e-8) -> int:
    return (torch.linalg.svdvals(X) > tol).sum().item()


@torch.jit.script
def _smi(
    X,
    Y,
    n_components: Optional[int] = 10,
    projection: str = "orthogonal",
    significance: bool = False,
    B: int = 10_000,
    center: bool = True,
):
    """
    Similarity of Matrices Index (SMI)

    A similarity index for comparing coupled data matrices.
    A two-step process starts with extraction of stable subspaces using Principal Component Analysis or some other
    method yielding two orthonormal bases. These bases are compared using Orthogonal Projection (OP / ordinary least
    squares) or Procrustes Rotation (PR). The result is a similarity measure that can be adjusted to various data sets
    and contexts and which includes explorative plotting and permutation based testing of matrix subspace equality.

    Reference: `A similarity index for comparing coupled matrices`_
    .. _A similarity index for comparing coupled matrices: https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.3049
    .. _Hoggorm implementation: https://github.com/olivertomic/hoggorm

    significance=True:
        Significance estimation for Similarity of Matrices Index (SMI)

        For each combination of components significance is estimated by sampling from a null distribution of no
        similarity, i.e. when the rows of one matrix is permuted B times and corresponding SMI values are computed. If
        the vector replicates is included, replicates will be kept together through permutations.
    """
    if center:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)

    rankX = _matrix_rank(X) if n_components is None else n_components
    rankY = _matrix_rank(Y) if n_components is None else n_components

    UX, _, _ = torch.linalg.svd(X)
    UY, _, _ = torch.linalg.svd(Y)

    m = torch.empty(())  # please torch.jit

    # Compute SMI values
    if projection == "orthogonal":
        m = (
            torch.arange(rankX, device=X.device)[:, None]
            .tile(1, rankX)
            .min(torch.arange(rankY, device=X.device)[None, :].tile(rankY, 1))
            .add(1)
            .reshape(rankX, rankY)
        )
        smi = (UX[:, :rankX].T @ UY[:, :rankY]).square().cumsum(1).cumsum(0) / m

    else:  # procrustes
        smi = torch.zeros((rankX, rankY), device=X.device)
        TU = UX[:, :rankX].T @ UY[:, :rankY]
        for p in range(rankX):
            for q in range(rankY):
                smi[p, q] = torch.linalg.svdvals(TU[: p + 1, : q + 1]).mean().square()

    # Recover wrong calculations (due to numerics)
    smi[smi > 1] = 1
    smi[smi < 0] = 0

    P = torch.zeros((rankX, rankY), device=X.device)

    if significance:
        BUX = UX.clone()

        if projection == "orthogonal":
            for __ in range(B):
                BUX = BUX[torch.randperm(len(BUX), device=X.device)]
                smiB = (BUX[:, :rankX].T @ UY[:, :rankY]).square().cumsum(1).cumsum(0) / m
                P[smi > torch.maximum(smiB, 1 - smiB)] += 1  # Increase P-value if non-significant permutation

        else:  # procrustes
            for __ in range(B):
                BUX = BUX[torch.randperm(len(BUX), device=X.device)]
                smiB = torch.zeros((rankX, rankY), device=X.device)
                TU = BUX[:, :rankX].T @ UY[:, :rankY]
                for p in range(rankX):
                    for q in range(rankY):
                        smiB[p, q] = torch.linalg.svdvals(TU[: p + 1, : q + 1]).mean().square()
                P[smi > torch.maximum(smiB, 1 - smiB)] += 1  # Increase P-value if non-significant permutation

        return smi, P / B

    return smi, -torch.ones(())


@torch.jit.script
def _r1(X, Y, center: bool = True):
    if center:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
    return torch.trace(X @ Y.T) / torch.sqrt(torch.trace(X @ X.T) * torch.trace(Y @ Y.T))


@torch.jit.script
def _r2(X, Y, center: bool = True):
    if center:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
    UX, sX, _ = torch.svd(X)
    UY, sY, _ = torch.svd(Y)
    return _r1(UX @ torch.diag(sX), UY @ torch.diag(sY))


@torch.jit.script
def _r3(X, Y, center: bool = True):
    if center:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
    UX, _, VX = torch.svd(X)
    UY, _, VY = torch.svd(Y)
    return _r1(UX @ VX.T, UY @ VY.T)


@torch.jit.script
def _r4(X, Y, center: bool = True):
    if center:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
    return _r1(torch.svd(X).U, torch.svd(Y).U)


@torch.jit.script
def _rG(X, Y, n_components: Optional[int] = None, center: bool = True):
    if center:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
    ncomp1 = _matrix_rank(X) if n_components is None else n_components
    ncomp2 = _matrix_rank(Y) if n_components is None else n_components
    UX = torch.svd(X).U[:, :ncomp1]
    UY = torch.svd(Y).U[:, :ncomp2]
    return _r1(UX.T @ UX, UY.T @ UY)


def _coxhead(X, Y, weighting: str = "sqrt"):
    s = math.sqrt(X.shape[1] * Y.shape[1]) if weighting == "sqrt" else float(min(X.shape[1], Y.shape[1]))
    Xcov, Ycov, diag = canonical_correlation_analysis(X, Y, backend="svd")
    cor = torch.diag(_pearson_correlation(Xcov, Ycov))
    C = 1 - s / torch.sum(1 / (1 - cor))
    return C


@torch.jit.script
def _coxhead2(X, Y, center: bool = True):
    if center:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
    if Y.shape[1] > X.shape[1]:
        Xt = Y
        Y = X
        X = Xt
    Rxx = X @ X.T
    Ryy = Y @ Y.T
    Rxy = X @ Y.T
    Ryx = Y @ X.T
    # Rxxinv = torch.linalg.solve(Rxx, torch.eye(len(Rxx), device="cuda"))
    Rxxinv = torch.linalg.pinv(Rxx)
    # RRRRinv = torch.linalg.solve(Ryy - Ryx @ Rxxinv @ Rxy, torch.eye(len(Ryy), device="cuda"))
    RRRRinv = torch.linalg.pinv(Ryy - Ryx @ Rxxinv @ Rxy)
    return torch.trace(RRRRinv @ Ryx @ Rxxinv @ Rxy) / torch.trace(RRRRinv @ Ryy)


def pearson(X, Y):
    return _pearson_correlation(X, Y).median()


def spearman(X, Y):
    return _spearman_correlation(X, Y).median()


def concordance(X, Y):
    return _concordance_correlation(X, Y).median()


def autocorrcorr(X, Y):
    return _autocorrelation_correlation(X, Y)


def rv(X, Y):
    return _rv([X, Y], modified=False)


def rv2(X, Y):
    return _rv([X, Y])


def smi(X, Y):
    return _smi(X, Y)[0].median()


def r1(X, Y):
    return _r1(X, Y)


def r3(X, Y):
    return _r3(X, Y)


def svcca(X, Y):
    return 1 - svcca_distance(X, Y, accept_rate=0.99, backend="svd")


def pwcca(X, Y):
    return 1 - pwcca_distance(X, Y, backend="svd")


def lcka(X, Y):
    return 1 - linear_cka_distance(X, Y, reduce_bias=False)


def op(X, Y):
    return 1 - orthogonal_procrustes_distance(X, Y)


if __name__ == "__main__":
    X = np.random.randn(400, 64)
    X = X - X.mean()
    U, s, V = np.linalg.svd(X, 0)
    Y1 = np.dot(np.dot(np.delete(U, 2, 1), np.diag(np.delete(s, 2))), np.delete(V, 2, 0))
    X, Y1 = torch.from_numpy(X).float().cuda(), torch.from_numpy(Y1).float().cuda()
    Y2 = torch.randn(400, 64, device="cuda")

    Y1, Y2 = Y1[:, :48], Y2[:, :48]

    for correlation in [pearson, spearman, concordance, autocorrcorr, rv, rv2, smi, r1, r3, svcca, pwcca, lcka, op]:
        try:
            print(
                f"{correlation.__name__}".ljust(12),
                f"correlated: {correlation(X, Y1).item():.4f}".ljust(20),
                f"3quarter: {correlation(X, 0.75*Y1+0.25*Y2).item():.4f}".ljust(20),
                f"half: {correlation(X, 0.5*Y1+0.5*Y2).item():.4f}".ljust(20),
                f"quarter: {correlation(X, 0.25*Y1+0.75*Y2).item():.4f}".ljust(20),
                f"random: {correlation(X, Y2).item():.4f}",
            )
        except:
            print(f"{correlation.__name__}")
