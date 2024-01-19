from typing import Union, Tuple, Collection
import numpy as np
import torch


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    assert len(perm.shape) == 1, "The perm must be a 1-d array"
    inv_perm = np.zeros_like(perm)
    inv_perm[perm] = np.arange(perm.shape[0])
    return inv_perm


def random_orthogonal(dim: int) -> np.ndarray:
    m = np.random.rand(dim, dim)
    q, r = np.linalg.qr(m)
    return q


def quantize(x: torch.Tensor, n_bins: int) -> np.ndarray:
    bin_size = (x.max() - x.min()) / n_bins
    return torch.round((x - x.min()) / bin_size) * bin_size + x.min()


def random_vec_with_seed(seed, size: Union[int, Collection[int]], range: Tuple[int, int]):
    random_generator = np.random.default_rng(seed)
    random_vec = random_generator.uniform(range[0], range[1], size)
    return random_vec


def generate_random_transformations(batch_size: int, n_random_vectors: int, ensure_sum_one: bool = True, dtype: torch.dtype = torch.float, device: str = "cpu"):
    inverse_transformation = torch.normal(0, 10, [batch_size, n_random_vectors, n_random_vectors], dtype=dtype, device=device)
    if ensure_sum_one:
        inverse_transformation /= torch.sum(inverse_transformation, dim=1, keepdim=True)
    transformation = torch.linalg.inv(inverse_transformation)
    return transformation, inverse_transformation


def generate_random_linear_combination(xs: torch.Tensor, n_random_vectors: int, transformation: torch.Tensor):
    """
    xs: [batch, n', dim], where n' < n_random_vectors
    Represent a vector by a linear combination of multiple vectors with different coefficients.
    """
    scale = xs.abs().max()
    if xs.shape[1] < n_random_vectors:
        xs = torch.cat([xs, 2 * scale * torch.rand(xs.shape[0], n_random_vectors - xs.shape[1], xs.shape[2], dtype=xs.dtype, device=xs.device) - scale], dim=1)
    rand_vecs = torch.bmm(transformation, xs)  # batched matarix multiplication
    return rand_vecs

def reconstruct_random_linear_combination(linear_combinations: torch.Tensor, inverse_transformation: torch.Tensor):
    """
    linear_combinations: [batch, n_random_vectors, dim]
    """
    reconstructed_vecs = torch.bmm(inverse_transformation[:, :1, :], linear_combinations)  # batched matarix multiplication
    return reconstructed_vecs



if __name__ == "__main__":
    def test_generate_random_linear_combination():
        transformation, inverse_transformation = generate_random_transformations(1, 3)
        xs = torch.tensor([[[8964., 1926, 817, 1, 2, 3, 4, 5]]])
        vecs = generate_random_linear_combination(xs, 3, transformation)
        reconstructed = reconstruct_random_linear_combination(vecs, inverse_transformation)

        print(reconstructed)
    

    test_generate_random_linear_combination()