from typing import Union, Tuple, Collection
import numpy as np
import torch
from torch import nn


from split_llm.common.utils import test_func


def copy_param(param_from: nn.Parameter, param_to: nn.Parameter):
    param_to.data = param_from.data.detach().clone()


def inverse_permutation(permutation: torch.Tensor):
    if len(permutation.shape) == 1:
        inv_perm = torch.zeros_like(permutation)
        inv_perm[permutation] = torch.arange(permutation.shape[0])
    elif len(permutation.shape) == 2:
        inv_perm = torch.zeros_like(permutation)
        inv_perm.scatter_(1, permutation, torch.arange(permutation.shape[1]).view(1, -1).expand(*permutation.shape))
    else:
        raise ValueError(f"Permutation can only be 1-d or 2-d, but got shape {permutation}")

    return inv_perm


def permute_2d(x: torch.Tensor, permutation: torch.Tensor):
    permuted_x = torch.zeros_like(x)
    permuted_x.scatter_(1, permutation, x)
    return permuted_x


if __name__ == "__main__":
    
    @test_func
    def test__inverse_permutation():
        permutation = torch.tensor([[3, 2, 1, 0], [2, 3, 1, 0]])
        inv_perm = inverse_permutation(permutation)
        print(inv_perm)

    @test_func
    def test__permute_2d():
        values = torch.tensor([[19, 26, 8, 17], [4, 8, 15, 16]])
        perms = torch.stack([torch.randperm(4) for _ in range(2)])
        permuted_values = permute_2d(values, perms)

        print("Permuted:", permuted_values)

        recovered_values = permute_2d(permuted_values, inverse_permutation(perms))
        print("Inversed:", recovered_values)

    test__inverse_permutation()
    test__permute_2d()