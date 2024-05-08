from typing import Union, Tuple, Collection
import numpy as np
import torch
from torch import nn


from perm_llm.common.utils import test_func


def copy_param(param_from: nn.Parameter, param_to: nn.Parameter):
    param_to.data = param_from.data.detach().clone()


def inverse_permutation(permutation: torch.Tensor):
    if len(permutation.shape) == 1:
        inv_perm = torch.zeros_like(permutation)
        inv_perm[permutation] = torch.arange(permutation.shape[0], device=permutation.device)
    elif len(permutation.shape) == 2:
        inv_perm = torch.zeros_like(permutation)
        inv_perm.scatter_(1, permutation, torch.arange(permutation.shape[1], device=permutation.device).view(1, -1).expand(*permutation.shape))
    else:
        raise ValueError(f"Permutation can only be 1-d or 2-d, but got shape {permutation}")

    return inv_perm


def permute_2d(x: torch.Tensor, permutation: torch.Tensor):
    permuted_x = torch.zeros_like(x)
    permuted_x.scatter_(1, permutation, x)
    return permuted_x


def relative_error(x: torch.Tensor, ref: torch.Tensor):
    if x.shape != ref.shape:
        raise ValueError("x must have the same shape with ref")
    return (torch.sqrt(torch.mean(torch.square(x - ref))) / torch.std(ref)).item()


def permute_with_seed(xs: torch.Tensor, seed: int, reverse: bool = False):
    raw_shape = xs.shape
    xs = xs.view(-1)
    rand_g = torch.Generator(device=xs.device).manual_seed(seed)
    permutation = torch.randperm(len(xs), generator=rand_g, device=xs.device)
    if reverse:
        permutation = inverse_permutation(permutation)

    return xs[permutation].view(*raw_shape)


def permute_2d_with_seed(xs: torch.Tensor, seed: int, reverse: bool = False):
    """
    scores: [q_len * n_heads * batch, k_len]
    """
    n_lists, k_len = xs.shape
    rand_g = torch.Generator(device=xs.device).manual_seed(seed)
    
    perm_list_level = torch.randperm(n_lists, generator=rand_g, device=xs.device)
    if reverse:
        perm_list_level = inverse_permutation(perm_list_level)
    inlist_perms = torch.stack([torch.randperm(k_len, generator=rand_g, device=xs.device) for _ in range(xs.shape[0])])
    if reverse:
        inlist_perms = inverse_permutation(inlist_perms)


    if not reverse:
        permuted_scores = xs.gather(1, inlist_perms)
        permuted_scores = permuted_scores[perm_list_level]
    else:
        permuted_scores = xs[perm_list_level]
        permuted_scores = permuted_scores.gather(1, inlist_perms)
        
    return permuted_scores



if __name__ == "__main__":
    
    @test_func
    def test__inverse_permutation():
        permutation = torch.tensor([[3, 2, 1, 0], [2, 3, 1, 0]])
        inv_perm = inverse_permutation(permutation)
        print(inv_perm)

    @test_func
    def test__permute_with_seed():
        seed = 1998
        xs = torch.rand([100, 100])
        permuted_xs = permute_with_seed(xs, seed)

    @test_func
    def test__permute_2d():
        values = torch.tensor([[19, 26, 8, 17], [4, 8, 15, 16]])
        perms = torch.stack([torch.randperm(4) for _ in range(2)])
        permuted_values = permute_2d(values, perms)

        print("Permuted:", permuted_values)

        recovered_values = permute_2d(permuted_values, inverse_permutation(perms))
        print("Inversed:", recovered_values)

    @test_func
    def test__permute_2d_with_seed():
        values = torch.tensor([[19, 26, 8, 17], [4, 8, 15, 16]])
        permuted_values = permute_2d_with_seed(values, 63)
        print("Permuted:", permuted_values)
        recovered_values = permute_2d_with_seed(permuted_values, 63, reverse=True)
        print("Inversed:", recovered_values)


    test__inverse_permutation()
    test__permute_2d()
    test__permute_2d_with_seed()