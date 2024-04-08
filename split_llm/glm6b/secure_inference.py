from typing import List, Callable

from ast import NodeTransformer
from split_llm.common.communication import Node
from split_llm.protocols.base_protocols import SS_Mul__CX_N0_Y_N1, SS_Mul__CX_N0, SS_Perm
from split_llm.protocols.ss_mul_with_memory import SS_Mul__AppendingX
from split_llm.protocols.element_wise import SS_ElementWise__RandPerm
from split_llm.glm6b.wrapped_layer import Attention_GLM_Wrapped




def setup_node(node: Node, attention_layers: List[Attention_GLM_Wrapped]):
    node.space.attentions = attention_layers


class GLMAttentionProtocol:
    def __init__(self, n0: Node, n1: Node, n2: Node, layer: int, mask_scale: float, max_generation_length: int = 500, device: str = "cpu"):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.layer = layer
    

        self.max_generation_length = max_generation_length

        self.device = device
        self.attn_protocols = [dict() for _ in range(28)]   # Totally 28 layers
    


        self.name = f"Attn_Layer_{layer}"

        self.qkv_mul_name = f"Attn_{layer}__qkv_matmul"
        self.dot_product_name = f"Attn_{layer}__dot_product"
        self.softmax_name = f"Attn_{layer}_softmax"


        self.qkv_mul_protocol = SS_Mul__CX_N0_Y_N1(
            [4096, 4096 * 3], 
            (lambda x, y: y @ x), self.qkv_mul_name,
            self.n0, self.n1, self.n2, mask_scale, self.device)
        
        self.dot_product_protocol = SS_Mul__AppendingX(
            [self.max_generation_length, 1, 32, 128], 0, 
            (lambda k, q: self.n0.space.attentions[self.layer].generate_logit_scores(q, k)),
            self.dot_product_name, self.n0, self.n1, self.n2,
            self.mask_scale, self.device
        )

        self.softmax_protocol = 
        

    def prepare(self):
        for protocols in self.attn_protocols:
            for k in protocols:
                protocols[k].prepare()

    def offline_execute(self):
        pass
