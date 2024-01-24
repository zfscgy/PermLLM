import torch

from desi_llm.glm6b.obfuscated_layer import WrappedGLMBlock


class ComputationNode:
    def __init__(self, layer: WrappedGLMBlock) -> None:
        self.layer = layer
        self.previous_kvs = []

    def forward_pass(self, *args):
        return self.layer(*args)
