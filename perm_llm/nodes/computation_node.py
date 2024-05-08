import torch

from desi_llm.glm6b.obfuscated_layer import WrappedGLMBlock


class ComputationNode:
    def __init__(self, layer: WrappedGLMBlock) -> None:
        self.layer = layer
        self.previous_kv = None

    def forward_pass(self, qkv: torch.Tensor, attention_mask: torch.Tensor, residual: torch.Tensor):
        q, k, v = qkv
        if self.previous_kv is None:
            self.previous_kv = (k, v)
        else:
            previous_k, previous_v = self.previous_kv
            previous_k = torch.concat([previous_k, k], dim=0)
            previous_v = torch.concat([previous_v, v], dim=0)
            self.previous_kv = (previous_k, previous_v)
        k, v = self.previous_kv
        return self.layer((q, k, v), attention_mask, residual)

    def reset_cache(self):
        self.previous_kv = None
