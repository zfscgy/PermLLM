from typing import Union, Tuple, Collection
import numpy as np
import torch


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


def shard(xs: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xs: [..., dim]
    Shard a vector, so that it can be represented by a linear combination of multiple shards
    """
    scale = xs.abs().max()

    
    trans = []
    vecss = []
    for kk in [k // 2, k - k // 2]:
        shape = (*xs.shape[:-1], kk-1, xs.shape[-1])
        extended_xs = torch.cat([xs.view(*xs.shape[:-1], 1, xs.shape[-1]), 
                                 2 * scale * torch.rand(shape, dtype=xs.dtype, device=xs.device) - scale], dim=-2)
        inv_transformation = 10 * torch.rand((*xs.shape[:-1], kk, kk), dtype=xs.dtype, device=xs.device) - 5
        rand_vecs = torch.matmul(torch.linalg.inv(inv_transformation), extended_xs)  # batched matarix multiplication
        trans.append(inv_transformation[..., 0, :])
        vecss.append(rand_vecs)
    return torch.cat(vecss, dim=-2), torch.cat(trans, dim=-1)


def reconstruct_random_linear_combination(linear_combinations: torch.Tensor, inverse_transformation: torch.Tensor) -> torch.Tensor:
    """
    linear_combinations: [batch, n_random_vectors, dim]
    """
    reconstructed_vecs = torch.bmm(inverse_transformation[:, 0, :], linear_combinations)  # batched matarix multiplication
    return reconstructed_vecs


if __name__ == "__main__":
    def test_random_orthonormal():
        m = random_orthonormal(4096)
        print(m @ m.T)

    def test_shard():
        xs = torch.tensor([[8964., 1926, 817, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 8964, 817, 1926]])
        shards, coefs = shard(xs, 10)
        reconstructed = coefs @ shards
        print(reconstructed)

    # test_random_orthonormal()
    test_shard()