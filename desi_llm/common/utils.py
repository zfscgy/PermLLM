import numpy as np
import torch


def inverse_permutation(perm: np.ndarray):
    assert len(perm.shape) == 1, "The perm must be a 1-d array"
    inv_perm = np.zeros_like(perm)
    inv_perm[perm] = np.arange(perm.shape[0])
    return inv_perm


def random_orthogonal(dim: int):
    m = np.random.rand(dim, dim)
    q, r = np.linalg.qr(m)
    return q @ q.T


def quantize(x: torch.Tensor, n_bins: int):
    bin_size = (x.max() - x.min()) / n_bins
    return torch.round((x - x.min()) / bin_size) * bin_size + x.min()