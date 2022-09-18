import torch


def sync_agg(data, slices, aggregate=torch.mean, axis=-1, pad_slice=False):
    if pad_slice:
        slices += [len(data) - 1]

    shape = list(data.shape)
    agg_shape = list(shape)
    agg_shape[axis] = len(slices)
    data_agg = torch.empty(agg_shape, dtype=data.dtype, device=data.device)

    idx_in = [slice(None)] * data.ndim
    idx_agg = [slice(None)] * data_agg.ndim

    for i, segment in enumerate(slices):
        idx_in[axis] = segment
        idx_agg[axis] = i
        data_agg[tuple(idx_agg)] = aggregate(data[tuple(idx_in)], axis=axis)

    return data_agg
