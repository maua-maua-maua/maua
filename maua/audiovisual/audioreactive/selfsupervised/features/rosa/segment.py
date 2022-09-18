import numpy as np
import torch
from torch.nn.functional import interpolate, pad
from torch_geometric.utils import get_laplacian


def distance_matrix(x, p=2):  # pairwise distance of vectors
    y = x

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2) + 1e-8
    dist = dist ** (1 / p)

    return dist


def recurrence_matrix(data, k=None, width=1, sym=False, bandwidth=None):
    t = data.shape[0]
    data = data.flatten(1)

    if k is None:
        if t > 2 * width + 1:
            k = 2 * np.ceil(np.sqrt(t - 2 * width + 1))
        else:
            k = 2
    k = int(k)

    # Build the neighbor search object
    rec = distance_matrix(data)

    # Remove connections within width
    for d in range(-width + 1, width):
        torch.diagonal(rec, offset=d).fill_(0)
    rec = rec + (rec == 0).float() * 1e20

    # Retain only the top-k links per point
    smallest_k = torch.topk(rec, k, dim=0, largest=False)
    rec = torch.scatter(torch.zeros_like(rec), dim=0, index=smallest_k.indices, src=smallest_k.values)

    # symmetrize
    if sym:
        rec = rec.minimum(rec.T)

    if bandwidth is None:
        bandwidth = torch.median(rec.max(axis=1).values)

    # Set all the negatives back to 0
    # Negatives are temporarily inserted above to preserve the sparsity structure
    # of the matrix without corrupting the bandwidth calculations
    rec = rec * (1 - (rec < 0).float())  # set to zero
    rec = torch.exp(rec / (-1 * bandwidth))
    rec = rec * (1 - (rec >= 1).float())  # set to zero

    return rec


def median_filter1d(x, k: int = 3, s: int = 1, p: int = 1):
    x = pad(x.unsqueeze(0), (p, p, 0, 0), mode="reflect").squeeze(0)
    x = x.unfold(1, k, s)
    x = x.median(dim=-1).values
    return x


def shear(X, factor):
    X_shear = torch.empty_like(X)
    for i in range(X.shape[1]):
        X_shear[:, i] = torch.roll(X[:, i], factor * i)
    return X_shear


def timelag_median_filter(rec):
    t = rec.shape[0]
    rec = pad(rec, (0, 0, 0, rec.shape[0]), mode="constant")
    lag = shear(rec, factor=-1)
    lag = median_filter1d(lag, k=7, s=1, p=3)
    rec = shear(lag, factor=1)
    rec = rec[:t]
    return rec


def init_plus_plus(ds, k):
    """
    Create cluster centroids using the k-means++ algorithm.
    From https://www.kdnuggets.com/2020/06/centroid-initialization-k-means-clustering.html
    """
    centroids = [ds[0]]
    for idx in range(1, k):
        dist_sq = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in ds])
        probs = dist_sq / (dist_sq.sum() + 1e-8)
        cumulative_probs = probs.cumsum()
        r = np.random.RandomState(42 + idx).rand()
        i = len(cumulative_probs) - 1
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        centroids.append(ds[i])
    return np.array(centroids)


def differentiable_k_means(data, k, num_iter, cluster_temp=5):
    """
    pytorch (differentiable) implementation of soft k-means clustering.
    https://github.com/bwilder0/clusternet/blob/master/models.py
    """
    # normalize x so it lies on the unit sphere
    data = torch.diag(1.0 / torch.norm(data, p=2, dim=1)) @ data
    # use kmeans++ initialization
    mu = torch.tensor(init_plus_plus(data.cpu().detach().numpy(), k), requires_grad=True).to(data)
    for _ in range(num_iter):
        # get distances between all data points and cluster centers
        dist = data @ mu.t()
        # cluster responsibilities via softmax
        r = torch.softmax(cluster_temp * dist, 1)
        # total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        # mean of points in each cluster weighted by responsibility
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        # update cluster means
        new_mu = torch.diag(1 / cluster_r) @ cluster_mean
        mu = new_mu
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp * dist, 1)
    return mu, r, dist


def laplacian_segmentation(envelope, beats, ks=[2, 4, 6, 8, 12, 16]):
    """Differentiable segmentation based on graph laplacian of weighted recurrence/sequence matrices

    Args:
        envelope (torch.Tensor): Feature to base segmentation on
        beats (List[int]): List of indices in envelope that are beats, this is used to reduce dimensionality of full
                           envelope to a beat-synchronous envelope (i.e. beat every 12 frames on average => 12x
                           reduction of computation)
        ks (list, optional): List of number of segments to return segmentations for. Defaults to [2, 4, 6, 8, 12, 16].

    Returns:
        List[torch.Tensor]: List of soft one-hot segmentations. The probabilities of segment membership per frame.
    """

    # make envelope beat-synchronous to reduce dimensionality
    Csync = []
    for beat1, beat2 in zip([0] + beats, beats + [len(envelope)]):
        Csync.append(torch.median(envelope[beat1:beat2], dim=0).values)
    Csync = torch.stack(Csync, dim=0)

    # build a weighted recurrence matrix using beat-synchronous envelope
    R = recurrence_matrix(Csync, width=3, sym=True)
    Rf = timelag_median_filter(R)

    # build the sequence matrix using envelope-similarity
    path_distance = torch.sum(torch.diff(Csync, dim=0) ** 2, dim=1)
    sigma = torch.median(path_distance)
    path_sim = torch.exp(-path_distance / sigma)
    R_path = torch.diag(path_sim, diagonal=1) + torch.diag(path_sim, diagonal=-1)

    # compute the balanced combination
    deg_path = torch.sum(R_path, dim=1)
    deg_rec = torch.sum(Rf, dim=1)
    mu = deg_path.dot(deg_path + deg_rec) / torch.sum((deg_path + deg_rec) ** 2)

    A = mu * Rf + (1 - mu) * R_path
    edge_index = A.nonzero()
    edge_weight = A[A.nonzero(as_tuple=True)]
    # compute the normalized laplacian and its spectral decomposition
    edge_index, edge_weight = get_laplacian(edge_index.T, edge_weight, normalization="sym")
    L = torch.sparse_coo_tensor(edge_index, edge_weight, device=A.device).to_dense()
    try:
        _, evecs = torch.linalg.eigh(L)
    except:
        _, evecs = torch.linalg.eig(L)
        evecs = evecs.real
    # median filter to smooth over small discontinuities
    evecs = median_filter1d(evecs.T, k=9, s=1, p=4).T
    # cumulative normalization for symmetric normalized laplacian eigenvectors
    Cnorm = torch.cumsum(evecs**2, dim=1) ** 0.5

    segmentations = []
    for k in ks:
        X = evecs[:, :k] / Cnorm[:, k - 1 : k]
        _, segmentation, _ = differentiable_k_means(data=X, k=k, num_iter=100)
        segmentation = interpolate(segmentation.T[None], size=envelope.shape[0], mode="nearest").squeeze().T
        segmentations.append(segmentation)
    return segmentations


import librosa as rosa
import scipy
import sklearn

BINS_PER_OCTAVE = 12 * 3
N_OCTAVES = 7


def laplacian_segmentation_rosa(audio, sr, out_size, ks=[2, 4, 6, 8, 16]):
    """
    Segments the audio with pattern recurrence analysis
    From https://librosa.org/doc/latest/auto_examples/plot_segmentation.html#sphx-glr-auto-examples-plot-segmentation-py
    """
    C = rosa.amplitude_to_db(
        np.abs(
            rosa.cqt(
                y=audio, sr=sr, hop_length=1024, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE
            )
        ),
        ref=np.max,
    )

    # make CQT beat-synchronous to reduce dimensionality
    _, beats = rosa.beat.beat_track(y=audio, sr=sr, trim=False, hop_length=1024)
    Csync = rosa.util.sync(C, beats, aggregate=np.median)

    # build a weighted recurrence matrix using beat-synchronous CQT
    R = rosa.segment.recurrence_matrix(Csync, width=3, mode="affinity", sym=True)
    # enhance diagonals with a median filter
    df = rosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))

    # build the sequence matrix using mfcc-similarity
    mfcc = rosa.feature.mfcc(y=audio, sr=sr, hop_length=1024)
    Msync = rosa.util.sync(mfcc, beats)
    path_distance = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    # compute the balanced combination
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec) ** 2)

    A = mu * Rf + (1 - mu) * R_path
    # compute the normalized laplacian and its spectral decomposition
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    _, evecs = scipy.linalg.eigh(L)
    # median filter to smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    # cumulative normalization for symmetric normalized laplacian eigenvectors
    Cnorm = np.cumsum(evecs**2, axis=1) ** 0.5

    segmentations = []
    for k in ks:
        X = evecs[:, :k] / Cnorm[:, k - 1 : k]
        segmentation = torch.from_numpy(sklearn.cluster.KMeans(n_clusters=k).fit_predict(X)).float()
        segmentation = interpolate(segmentation[None, None, :], size=out_size, mode="nearest").squeeze()
        segmentations.append(segmentation)
    return torch.stack(segmentations, dim=1).long()


if __name__ == "__main__":
    envelope = torch.randn((128, 12))
    beats = [4, 7, 14, 20, 25, 36, 47, 50, 57, 67, 70, 77, 82, 88, 92, 97, 100, 107, 117]

    Rr = rosa.segment.recurrence_matrix(envelope.T.numpy(), width=3, sym=True, mode="affinity")
    Rt = recurrence_matrix(envelope, width=3, sym=True)

    Rrf = rosa.segment.timelag_filter(scipy.ndimage.median_filter)(Rr, size=(1, 7))
    Rtf = timelag_median_filter(Rt)

    Cr = laplacian_segmentation_rosa(envelope.T.numpy(), beats)
    Ct = laplacian_segmentation(envelope, beats)
