import torch


def sqrtm(matrix, num_iters=100):
    """
    Square root of matrix using Newton-Schulz Iterative method
    Source: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py

    Params:
        matrix: Two dimensional tensor
        num_iters: Maximum number of iterations
    Returns:
        Square root of matrix
    """
    dtype = matrix.dtype
    matrix = matrix.double()

    expected_num_dims = 2
    if matrix.dim() != expected_num_dims:
        raise ValueError(f"Input dimension equals {matrix.dim()}, expected {expected_num_dims}")

    if num_iters <= 0:
        raise ValueError(f"Number of iteration equals {num_iters}, expected greater than 0")

    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p="fro")
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, requires_grad=False).to(matrix)
    Z = torch.eye(dim, dim, requires_grad=False).to(matrix)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1).to(matrix)

    prev_error = 1_000_000
    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = torch.norm(matrix - torch.mm(s_matrix, s_matrix)) / torch.norm(matrix)
        if torch.isclose(error, torch.zeros_like(error), atol=1e-5) or error > prev_error:  # guard against divergence
            break
        prev_error = error

    return s_matrix.to(dtype)


def symsqrtm(matrix):
    """Compute the square root of a positive definite matrix."""
    # perform the decomposition
    _, s, v = matrix.svd()
    # truncate small components
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    # compose the square root matrix
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def frechet_distance(feats1, feats2, eps=1e-6):
    """
    Torch implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is:
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    Original stable version by Danica J. Sutherland. Adapted for PyTorch by Hans Brouwer
    Params:
        feats1: Tensor containing the activation features of generated samples
        feats2: Tensor containing the activation features of a representative data set
    Returns:
        Frechet distance between feature sets
    """
    mu1, sigma1 = torch.mean(feats1, axis=0), torch.cov(feats1.T)
    mu2, sigma2 = torch.mean(feats2, axis=0), torch.cov(feats2.T)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)

    if not torch.isfinite(covmean).all():  # Product might be almost singular
        print(f"Frechet distance calculation produced singular product; adding {eps} to diagonal of cov estimates")
        offset = torch.eye(sigma1.shape[0], device=feats1.device) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if torch.is_complex(covmean):  # Numerical error might give slight imaginary component
        if not torch.allclose(torch.diagonal(covmean).imag, 0, atol=1e-3):
            m = torch.max(torch.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    distance = diff @ diff + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)
    return distance.item()
