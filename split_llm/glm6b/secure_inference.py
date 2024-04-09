from typing import List, Callable
from functools import partial

import numpy as np

import torch

from split_llm.common.communication import Node
from split_llm.protocols.base_protocols import SS_Mul__CX_N0_Y_N1, SS_Mul__CX_N0, SS_Perm
from split_llm.protocols.ss_mul_with_memory import SS_Mul__AppendingX
from split_llm.protocols.element_wise import SS_ElementWise__RandPerm
from split_llm.glm6b.wrapped_layer import Attention_GLM_Wrapped
from split_llm.glm6b.utils import permute_attention_scores_with_seed



def setup_node(node: Node, attention_layers: List[Attention_GLM_Wrapped]):
    node.space.attentions = attention_layers


class GLMAttentionProtocol:
    def __init__(self, n0: Node, n1: Node, n2: Node, layer: int, mask_scale: float, max_generation_length: int = 500, device: str = "cpu"):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.layer = layer
    

        self.max_generation_length = max_generation_length
        self.mask_scale = mask_scale

        self.device = device
    

        self.name = f"Attn_Layer_{layer}"

        self.qkv_mul_name = f"Attn_{layer}__qkv_matmul"
        self.dot_product_name = f"Attn_{layer}__dot_product"
        self.softmax_name = f"Attn_{layer}_softmax"
        self.weighted_sum_name = f"Attn_{layer}_weighted_sum"


        self.qkv_mul_protocol = SS_Mul__CX_N0(
            [4096, 4096 * 3], 
            (lambda x, y: y @ x), self.qkv_mul_name,
            self.n0, self.n1, self.n2, 
            mask_scale, device)
        
        self.dot_product_protocol = SS_Mul__AppendingX(
            [self.max_generation_length, 1, 32, 128], 0, 
            (lambda k, q: self.n0.space.attentions[self.layer].generate_logit_scores(q, k)),
            self.dot_product_name, self.n0, self.n1, self.n2,
            mask_scale, device
        )

        self.softmax_protocol = SS_ElementWise__RandPerm(
            permute_attention_scores_with_seed, partial(permute_attention_scores_with_seed, reversed=True),
            partial(torch.softmax, dim=-1), self.softmax_name,
            self.n0, self.n1, self.n2, 
            mask_scale, device
        )

        self.weighted_sum_protocol = SS_Mul__AppendingX(
            [self.max_generation_length, 1, 32, 128], 0,
            (lambda v, score: self.n0.space.attentions[self.layer].generate_weighted_values(score, v)),
            self.weighted_sum_name,
            self.n0, self.n1, self.n2,
            mask_scale, device
        )

        self.n0.storage[f"{self.qkv_mul_name}:x"] = self.n0.space.attentions[self.layer].qkv_weight.T

    def prepare(self):
        self.qkv_mul_protocol.prepare()
        self.dot_product_protocol.prepare()
        self.softmax_protocol.prepare()
        self.weighted_sum_protocol.prepare()

    def offline_execute(self, next_length: int, total_length: int):
        self.qkv_mul_protocol.offline_execute([next_length, 1, 4096], [next_length, 1, 4096 * 3])
        self.dot_product_protocol.offline_execute([next_length, 1, 32, 128], [next_length, total_length, 1, 32], next_length)

        perm_key = np.random.randint(2 ** 30)
        self.n0.storage[f"{self.softmax_name}:new_perm"] = perm_key
        self.n0.storage[f"{self.softmax_name}:new_invperm"] = perm_key
        self.softmax_protocol.offline_execute([next_length, total_length, 1, 32])
        self.weighted_sum_protocol.offline_execute([total_length, 1, 32, 128], [next_length, total_length, 32, 128], next_length)
    
    def online_step_qkv(self):
        """
        Input:
            Node 0: x0
            Node 1: x1
        """
        self.n0.storage[f"{self.qkv_mul_name}:y0"] = self.n0.storage[f"{self.name}:x0"]
        self.n1.storage[f"{self.qkv_mul_name}:y1"] = self.n1.storage[f"{self.name}:x1"]

        self.qkv_mul_protocol.online_execute()

        self.n0.storage[f"{self.name}:h0"] = self.n0.storage[f"{self.qkv_mul_name}:z0"]
        self.n1.storage[f"{self.name}:h1"] = self.n1.storage[f"{self.qkv_mul_name}:z1"]

        self.qkv_mul_protocol.clear_io()
    
    def online_step_dot_product(self):
        pass