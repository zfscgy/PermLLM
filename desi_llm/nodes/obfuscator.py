from typing import Callable

import numpy as np
import torch
from torch import nn

from desi_llm.glm6b.configs import GLM6BConfig
from desi_llm.glm6b.obfuscated_layer import obfuscate_transformer, generate_obfuscation_keys, ObfuscationKeys, WrappedGLMBlock, expand_segmented_keys


class ObfuscatorNode:
    def __init__(self) -> None:
        self.key = None
    
    def obfuscate(self, block: WrappedGLMBlock, device: str="cpu") -> WrappedGLMBlock:
        """
        Generate the obfuscated block (in-place)
        """
        self.key = generate_obfuscation_keys(GLM6BConfig.model_dim, GLM6BConfig.n_attention_heads, device)
        return obfuscate_transformer(block, self.key, device)
    
    def forward_pass(self, x, key_name: str, reverse: bool = False):
        """
        x: [..., model_dim]
        """
        if key_name == "v":
            key: torch.Tensor = self.key.qkv[2]
        else:
            key: torch.Tensor = getattr(self.key, key_name, None)
        if key is None:
            raise AssertionError(f'Key name {key} is not found')
        if isinstance(key, tuple) or isinstance(key, list):  # QKV case
            if key_name == "qkv":
                q_key, k_key, v_key = map(expand_segmented_keys, [k[0] for k in key], [k[1] for k in key])
                q, k, v = x
                n_heads, head_dim = q.shape[-2:]
                q, k, v = q.view(*q.shape[:-2], -1), k.view(*k.shape[:-2], -1), v.reshape(*v.shape[:-2], -1)
                if reverse:
                    q1, k1, v1 = q @ q_key, k @ k_key, v @ v_key
                else:
                    q1, k1, v1 = q @ q_key.T, k @ k_key.T, v @ v_key.T
                return q1.view(*q1.shape[:-1], n_heads, head_dim), k1.view(*k1.shape[:-1], n_heads, head_dim), v1.view(*v1.shape[:-1], n_heads, head_dim)
            else:  # only forward V
                v_key = expand_segmented_keys(*key)
                if reverse:
                    v1 = x @ v_key
                else:
                    v1 = x @ v_key.T
                return v1

        else:
            if reverse:
                return x @ key
            else:
                return x @ key.T