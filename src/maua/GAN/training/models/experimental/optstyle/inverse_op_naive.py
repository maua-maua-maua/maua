import torch


def inverse_conv(z, w, is_upper, dilation):
    batchsize, height, width, n_channels = z.shape
    ksize = w.shape[0]
    kcenter = (ksize - 1) // 2

    x = torch.zeros_like(z)

    def filter2image(j, i, m, k):
        m_ = (m - kcenter) * dilation
        k_ = (k - kcenter) * dilation
        return j + k_, i + m_

    def in_bound(idx, lower, upper):
        return (idx >= lower) and (idx < upper)

    def reverse_range(n, reverse):
        if reverse:
            return range(n)
        else:
            return reversed(range(n))

    for b in range(batchsize):
        for j in reverse_range(height, is_upper):
            for i in reverse_range(width, is_upper):
                for c_out in reverse_range(n_channels, not is_upper):
                    for c_in in range(n_channels):
                        for k in range(ksize):
                            for m in range(ksize):
                                if k == kcenter and m == kcenter and c_in == c_out:
                                    continue

                                j_, i_ = filter2image(j, i, m, k)

                                if not in_bound(j_, 0, height):
                                    continue

                                if not in_bound(i_, 0, width):
                                    continue

                                x[b, j, i, c_out] -= w[k, m, c_in, c_out] * x[b, j_, i_, c_in]

                    # Compute value for x
                    x[b, j, i, c_out] += z[b, j, i, c_out]
                    x[b, j, i, c_out] /= w[kcenter, kcenter, c_out, c_out]

    return x
