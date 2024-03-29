from typing import Union, Tuple, Collection
import numpy as np
import torch
from torch import nn


def inverse_permutation(perm: Union[np.ndarray, torch.Tensor], device: str="cpu") -> torch.Tensor:
    assert len(perm.shape) == 1, "The perm must be a 1-d array"
    if isinstance(perm, np.ndarray):
        perm = torch.tensor(perm, device=device)
    inv_perm = torch.zeros_like(perm)
    inv_perm[perm] = torch.arange(perm.shape[0], dtype=perm.dtype, device=perm.device)
    return inv_perm


def random_orthonormal_qr(dim: int, device: str="cpu") -> torch.Tensor:
    m = torch.normal(0, 1, [dim, dim], device=device)
    q, r = torch.linalg.qr(m)
    d = torch.diagonal(r)
    q *= d / torch.abs(d)
    return q


def random_orthonormal_household_permuted(dim: int, n: int=10, device: str="cpu")->torch.Tensor:
    m = torch.eye(dim, dim, device=device)[torch.randperm(dim, device=device)]
    def household():
        v = torch.normal(0, 1, [dim, 1], device=device)
        v = v / torch.norm(v)
        return (m - 2 * v @ v.T)[torch.randperm(dim, device=device)][:, torch.randperm(dim, device=device)]

    p = m
    for _ in range(n):
        p @= household()
    return p


random_orthonormal = random_orthonormal_household_permuted


def quantize(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    bin_size = (x.max() - x.min()) / n_bins
    return torch.round((x - x.min()) / bin_size) * bin_size + x.min()


def random_vec_with_seed(seed, size: Union[int, Collection[int]], range: Tuple[int, int]) -> np.ndarray:
    random_generator = np.random.default_rng(seed)
    random_vec = random_generator.uniform(range[0], range[1], size)
    return random_vec


def copy_param(param_from: nn.Parameter, param_to: nn.Parameter):
    param_to.data = param_from.data.detach().clone()


if __name__ == "__main__":
    def test_random_orthonormal():
        m = random_orthonormal(4096)
        print(m @ m.T)

    # test_random_orthonormal()