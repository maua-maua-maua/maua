import torch


def interpolate(x0, x1, num_midpoints):
    """Linear interpolation grid between x0 and x1 (vendored from StudioGAN src/utils/misc.py)."""
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device=x0.device).to(x0.dtype)
    return (x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1))


def apply_sefa(generator, backbone, z, fake_label, num_semantic_axis, maximum_variations, num_cols):
    w = generator.linear0.weight
    eigen_vectors = torch.svd(w).V.to(z.device)[:, :num_semantic_axis]
    z_dim = len(z)
    zs_start = z.repeat(num_semantic_axis).view(-1, 1, z_dim)
    zs_end = (z.unsqueeze(1) + maximum_variations * eigen_vectors).T.view(-1, 1, z_dim)
    zs_canvas = interpolate(x0=zs_start, x1=zs_end, num_midpoints=num_cols - 2).view(-1, zs_start.shape[-1])
    images_canvas = generator(zs_canvas, fake_label.repeat(len(zs_canvas)), eval=True)
    return images_canvas


def cff(ckpt_path, out_path):
    """Closed-form factorization of a (rosinality-format) StyleGAN2 checkpoint's style modulation weights."""
    ckpt = torch.load(ckpt_path)
    modulate = {k: v for k, v in ckpt["g_ema"].items() if "modulation" in k and "to_rgbs" not in k and "weight" in k}

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")

    torch.save({"ckpt": ckpt_path, "eigvec": eigvec}, out_path)
