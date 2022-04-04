"""
prdc 
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import numpy as np
import torch


def pairwise_distances(x, y=None):
    """
    Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    Input: x is a Nxd matrix y is an optional Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    """
    if y is None:
        y = x
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    x_norm = (x**2).sum(2).view(x.shape[0], x.shape[1], 1)
    y_t = y.permute(0, 2, 1).contiguous()
    y_norm = (y**2).sum(2).view(y.shape[0], 1, y.shape[1])
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    dist[dist != dist] = 0  # replace nan values with 0
    return np.array(torch.clamp(dist, 0.0, np.inf)[0])


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def nearest_neighbor_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = pairwise_distances(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def prdc(real_features, fake_features, nearest_k=5):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        precision, recall, density, and coverage.
    """

    real_nearest_neighbour_distances = nearest_neighbor_distances(real_features, nearest_k)
    fake_nearest_neighbour_distances = nearest_neighbor_distances(fake_features, nearest_k)
    distance_real_fake = pairwise_distances(real_features, fake_features)

    precision = (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()

    recall = (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()

    return precision, recall, density, coverage
