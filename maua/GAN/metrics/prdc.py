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
    return torch.clamp(dist, 0.0, np.inf)[0]


def nearest_neighbor_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = pairwise_distances(input_features)
    radii = torch.topk(distances, k=nearest_k + 1, largest=False, axis=-1).values.max(-1).values
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

    precision = (distance_real_fake < real_nearest_neighbour_distances[:, None]).any(0).float().mean()
    recall = (distance_real_fake < fake_nearest_neighbour_distances[None, :]).any(1).float().mean()
    density = (distance_real_fake < real_nearest_neighbour_distances[:, None]).sum(0).float().mean() / nearest_k
    coverage = (distance_real_fake.min(axis=1).values < real_nearest_neighbour_distances).float().mean()

    return precision.item(), recall.item(), density.item(), coverage.item()
