from typing import List, Callable

from ast import NodeTransformer
from split_llm.common.communication import Node
from split_llm.base_protocols import SS_Mul__CX_N0_Y_N1, SS_Mul__CX_N0, SS_Perm
from split_llm.glm6b.wrapped_layer import Attention_GLM_Wrapped



def setup_node(node: Node, attention_layers: List[Attention_GLM_Wrapped]):
    node.space.attentions = attention_layers


class SplitGLM:
    def __init__(self, n0: Node, n1: Node, n2: Node, device: str):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.device = device
        self.attn_protocols = [dict() for _ in range(28)]   # Totally 28 layers


    def buid_attention_forward_protocols(self, layer: int, mask_scale: float):
        ss_mul_qkv = SS_Mul__CX_N0_Y_N1(
            [4096, 4096 * 3], 
            (lambda x, y: y @ x), f"Attn_{layer}__qkv_matmul",
            self.n0, self.n1, self.n2, mask_scale, self.device)
        self.protocols[layer]["qkv"] = ss_mul_qkv
        
        # This protocol is used to compute the 

    def prepare(self):
        for protocols in self.attn_protocols:
            for k in protocols:
                protocols[k].prepare()

    def offline_computation(self, input_len: int):
        pass
