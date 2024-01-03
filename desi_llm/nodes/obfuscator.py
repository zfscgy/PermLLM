from typing import Callable

import numpy as np
from torch import nn

from desi_llm.glm6b.configs import GLM6BConfig
from desi_llm.glm6b.obfuscated_layer import obfuscate_transformer, generate_obfuscation_keys, ObfuscationKeys, WrappedGLMBlock


class ObfuscatorNode:
    def __init__(self) -> None:
        self.key = None
    
    def obfuscate(self, block: WrappedGLMBlock) -> WrappedGLMBlock:
        """
        Generate the obfuscated block (in-place)
        """
        self.key = generate_obfuscation_keys(GLM6BConfig.model_dim, GLM6BConfig.n_attention_heads)
        return obfuscate_transformer(block, self.key)
    
    def forward_pass(self, x: np.ndarray, key_name: str):
        key = getattr(self.key, key_name, None)
        if key is None:
            raise AssertionError(f'Key name {key} is not found')
        return key @ x
    
    def backward_pass(self, x: np.ndarray, key_name: str):
        key: np.ndarray = getattr(self.key, key_name, None)
        if key is None:
            raise AssertionError(f'Key name {key} is not found')
        return key.T @ x
    
