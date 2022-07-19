import torch
import torch.nn.functional as F


def normalize(img):
    out = img - img.min()
    out = out / out.max()
    return out


def random_rotation(N):
    """
    Draws random N-dimensional rotation matrix (det = 1, inverse = transpose) from the special orthogonal group
    From https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/stats/_multivariate.py#L3309
    """
    H = torch.eye(N)
    D = torch.empty((N,))
    for n in range(N - 1):
        x = torch.randn(N - n)
        norm2 = x @ x
        x0 = x[0].item()
        D[n] = torch.sign(x[0]) if x[0] != 0 else 1
        x[0] += D[n] * torch.sqrt(norm2)
        x /= torch.sqrt((norm2 - x0**2 + x[0] ** 2) / 2.0)
        H[:, n:] -= torch.outer(H[:, n:] @ x, x)
    D[-1] = (-1) ** (N - 1) * D[:-1].prod()
    H = (D * H.T).T
    return H


def sliced_optimal_transport(source, target, iters=8):
    source, target = source.permute(0, 2, 3, 1), target.permute(0, 2, 3, 1).to(source)  # [b, h, w, c]
    for it in range(iters):
        rotation = random_rotation(source.shape[-1]).to(source)
        rotated_output = source @ rotation
        rotated_style = target @ rotation
        matched_output = hist_match(rotated_output, rotated_style)
        source = matched_output @ rotation.T  # rotate back to normal
    source, target = source.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)  # [b, c, h, w]
    return source


def hist_match(target, source, mode="cdf", eps=1e-2):
    target = target.permute(0, 3, 1, 2)  # -> b, c, h, w
    source = source.permute(0, 3, 1, 2)

    if mode == "cdf":
        b, c, h, w = target.shape
        matched = cdf_match(target, source).clamp(target.min(), target.max()).to(target)

    else:
        # based on https://github.com/ProGamerGov/Neural-Tools/blob/master/linear-color-transfer.py#L36

        mu_t = target.mean((2, 3), keepdim=True)
        hist_t = (target - mu_t).view(target.size(1), -1)  # [c, b * h * w]
        cov_t = hist_t @ hist_t.T / hist_t.shape[1] + eps * torch.eye(hist_t.shape[0]).to(hist_t)

        mu_s = source.mean((2, 3), keepdim=True)
        hist_s = (source - mu_s).view(source.size(1), -1)
        cov_s = hist_s @ hist_s.T / hist_s.shape[1] + eps * torch.eye(hist_s.shape[0]).to(hist_s)

        if mode == "chol":
            chol_t = torch.linalg.cholesky(cov_t)
            chol_s = torch.linalg.cholesky(cov_s)
            matched = chol_s @ torch.inverse(chol_t) @ hist_t

        elif mode == "pca":
            eva_t, eve_t = torch.symeig(cov_t, eigenvectors=True, upper=True)
            Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
            eva_s, eve_s = torch.symeig(cov_s, eigenvectors=True, upper=True)
            Qs = eve_s @ torch.sqrt(torch.diag(eva_s)) @ eve_s.T
            matched = Qs @ torch.inverse(Qt) @ hist_t

        elif mode == "sym":
            eva_t, eve_t = torch.symeig(cov_t, eigenvectors=True, upper=True)
            Qt = eve_t @ torch.sqrt(torch.diag(eva_t)) @ eve_t.T
            Qt_Cs_Qt = Qt @ cov_s @ Qt
            eva_QtCsQt, eve_QtCsQt = torch.symeig(Qt_Cs_Qt, eigenvectors=True, upper=True)
            QtCsQt = eve_QtCsQt @ torch.sqrt(torch.diag(eva_QtCsQt)) @ eve_QtCsQt.T
            matched = torch.inverse(Qt) @ QtCsQt @ torch.inverse(Qt) @ hist_t

        matched = matched.view(*target.shape) + mu_s

    return matched.permute(0, 2, 3, 1)  # -> b, h, w, c


def interp(x, xp, fp):
    # based on https://github.com/numpy/numpy/blob/main/numpy/core/src/multiarray/compiled_base.c#L489

    f = torch.zeros_like(x)

    idxs = torch.searchsorted(xp, x)
    idxs_next = (idxs + 1).clamp(0, len(xp) - 1)

    slopes = (fp[idxs_next] - fp[idxs]) / (xp[idxs_next] - xp[idxs])
    f = slopes * (x - xp[idxs]) + fp[idxs]

    infinite = ~torch.isfinite(f)
    if infinite.any():
        inf_idxs_next = idxs_next[infinite]
        f[infinite] = slopes[infinite] * (x[infinite] - xp[inf_idxs_next]) + fp[inf_idxs_next]

        still_infinite = ~torch.isfinite(f)
        if still_infinite.any():
            f[still_infinite] = fp[idxs[still_infinite]]

    return f


def cdf_match(target, source, bins=256):
    B, C, H, W = target.shape
    target, source = target.reshape(C, -1).float(), source.reshape(C, -1).float()
    matched = torch.empty_like(target)
    for i, (target_channel, source_channel) in enumerate(zip(target.contiguous(), source)):
        lo = torch.min(target_channel.min(), source_channel.min())
        hi = torch.max(target_channel.max(), source_channel.max())

        # TODO find batched method of getting histogram? maybe based on numpy's impl?
        # https://github.com/numpy/numpy/blob/v1.20.0/numpy/lib/histograms.py#L678
        target_hist = torch.histc(target_channel, bins, lo, hi)
        source_hist = torch.histc(source_channel, bins, lo, hi)
        print(target_hist.shape, target_hist.min(), torch.median(target_hist), target_hist.max())
        print(source_hist.shape, source_hist.min(), torch.median(source_hist), source_hist.max())
        bin_edges = torch.linspace(lo, hi, bins + 1)[1:].to(target)
        plt.bar(bin_edges, target_hist, alpha=0.5)
        plt.bar(bin_edges, source_hist, alpha=0.5)
        plt.show()
        plt.close()

        target_cdf = target_hist.cumsum(0)
        target_cdf = target_cdf / target_cdf[-1]

        source_cdf = source_hist.cumsum(0)
        source_cdf = source_cdf / source_cdf[-1]

        remapped_cdf = interp(target_cdf, source_cdf, bin_edges)
        print(target_cdf.shape, target_cdf.min(), torch.median(target_cdf), target_cdf.max())
        print(source_cdf.shape, source_cdf.min(), torch.median(source_cdf), source_cdf.max())
        print(remapped_cdf.shape, remapped_cdf.min(), torch.median(remapped_cdf), remapped_cdf.max())
        print()
        matched[i] = interp(target_channel, bin_edges, remapped_cdf)
    return matched.reshape(B, C, H, W)


import matplotlib.pyplot as plt


def fast_cdf_match(target, source, bins=256):
    B, C, H, W = target.shape
    target, source = target.reshape(C, -1).float(), source.reshape(C, -1).float()
    matched = torch.empty_like(target)
    for i, (target_channel, source_channel) in enumerate(zip(target.contiguous(), source)):

        lo = torch.min(target_channel.min(), source_channel.min())
        hi = torch.max(target_channel.max(), source_channel.max())

        print(F.one_hot(((target_channel - lo) / (hi - lo)).mul(bins - 1).round().long()).shape)
        print(
            ((target_channel - lo) / (hi - lo)).min(),
            ((target_channel - lo) / (hi - lo)).mean(),
            ((target_channel - lo) / (hi - lo)).max(),
        )
        target_hist = F.one_hot(((target_channel - lo) / (hi - lo)).mul(bins - 1).round().long()).sum(-2)
        source_hist = F.one_hot(((source_channel - lo) / (hi - lo)).mul(bins - 1).round().long()).sum(-2)
        print(target_hist.shape, target_hist.min(), torch.median(target_hist), target_hist.max())
        print(source_hist.shape, source_hist.min(), torch.median(source_hist), source_hist.max())
        bin_edges = torch.linspace(lo, hi, bins + 1)[1:].to(target)
        plt.bar(bin_edges, target_hist, alpha=0.5)
        plt.bar(bin_edges, source_hist, alpha=0.5)
        plt.show()
        plt.close()

        target_cdf = target_hist.cumsum(0)
        target_cdf = target_cdf / target_cdf[-1]
        print(target_cdf.shape, target_cdf.min(), torch.median(target_cdf), target_cdf.max())

        source_cdf = source_hist.cumsum(0)
        source_cdf = source_cdf / source_cdf[-1]
        print(source_cdf.shape, source_cdf.min(), torch.median(source_cdf), source_cdf.max())

        remapped_cdf = interp(target_cdf, source_cdf, bin_edges)
        print(remapped_cdf.shape, remapped_cdf.min(), torch.median(remapped_cdf), remapped_cdf.max())
        print()
        matched[i] = interp(target_channel, bin_edges, remapped_cdf)
    return matched.reshape(B, C, H, W)


if __name__ == "__main__":
    from time import time

    trials = 1

    a = torch.randn(1, 3, 256, 384)
    b = torch.randn(1, 3, 256, 384)
    b = b - b.min()
    print(a.min(), torch.median(a), a.max())
    print(b.min(), torch.median(b), b.max())

    t = time()
    for _ in range(trials):
        c = cdf_match(a, b)
    print((time() - t) / trials)
    print(c.min(), torch.median(c), c.max())

    a = torch.randn(1, 3, 256, 384)
    b = torch.randn(1, 3, 256, 384)
    b = b - b.min()

    t = time()
    for _ in range(trials):
        c = fast_cdf_match(a, b)
    print((time() - t) / trials)
    print(c.min(), torch.median(c), c.max())
