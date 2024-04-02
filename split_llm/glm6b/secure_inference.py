from typing import List

from ast import NodeTransformer
from split_llm.common.communication import Node
from split_llm.protocols import SS_Mul__CX_N0_Y_N1, SS_Mul, SS_Perm
from split_llm.glm6b.wrapped_layer import Attention_GLM_Wrapped



def setup_node(node: Node, attention_layers: List[Attention_GLM_Wrapped])


class SplitLLM:
    def __init__(self, n0: Node, n1: Node, n2: Node, device: str):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.device = device

    def buid_attention_forward_protocols(self, layer: int):
        ss_mul_qkv = SS_Mul__CX_N0_Y_N1()