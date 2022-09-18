import torch
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from .features.processing import gaussian_filter


def spline_loop_latents(y, size, n_loops=1):
    y = torch.cat((y, y[[0]]))
    t_in = torch.linspace(0, 1, len(y)).to(y)
    t_out = torch.linspace(0, n_loops, size).to(y) % 1
    coeffs = natural_cubic_spline_coeffs(t_in, y.permute(1, 0, 2))
    out = NaturalCubicSpline(coeffs).evaluate(t_out)
    return out.permute(1, 0, 2)


def latent_patch(
    rng,
    latents,
    palette,
    segmentations,
    features,
    tempo,
    fps,
    patch_type,
    segments,
    loop_bars,
    seq_feat,
    seq_feat_weight,
    mod_feat,
    mod_feat_weight,
    merge_type,
    merge_depth,
):
    feature = seq_feat_weight * features[seq_feat]
    segmentation = segmentations[(seq_feat, segments)]
    permutation = torch.randperm(len(palette), generator=rng, device=rng.device)

    if patch_type == "segmentation":
        selection = permutation[:segments]
        selectseq = selection[segmentation.cpu().numpy()]
        sequence = palette[selectseq]
        sequence = gaussian_filter(sequence, 5)
    elif patch_type == "feature":
        n_select = feature.shape[1]
        if n_select == 1:
            selection = permutation[:2]
            sequence = feature[..., None] * palette[selection][[0]] + (1 - feature[..., None]) * palette[selection][[1]]
        else:
            selection = permutation[:n_select]
            sequence = torch.einsum("TN,NWL->TWL", feature, palette[selection])
    elif patch_type == "loop":
        selection = permutation[:segments]
        n_loops = len(latents) / fps / 60 / tempo / 4 / loop_bars
        sequence = spline_loop_latents(palette[selection], len(latents), n_loops=n_loops)
    sequence = gaussian_filter(sequence, 1)

    if merge_depth == "low":
        lays = slice(0, 6)
    elif merge_depth == "mid":
        lays = slice(6, 12)
    elif merge_depth == "high":
        lays = slice(12, 18)
    elif merge_depth == "lowmid":
        lays = slice(0, 12)
    elif merge_depth == "midhigh":
        lays = slice(6, 18)
    elif merge_depth == "all":
        lays = slice(0, 18)

    if merge_type == "average":
        latents[:, lays] += sequence[:, lays]
        latents[:, lays] /= 2
    elif merge_type == "modulate":
        modulation = mod_feat_weight * features[mod_feat][..., None]
        latents[:, lays] *= 1 - modulation
        latents[:, lays] += modulation * sequence[:, lays]
    else:  # overwrite
        latents[:, lays] = sequence[:, lays]

    return latents
