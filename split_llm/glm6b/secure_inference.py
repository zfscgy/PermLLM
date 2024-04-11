from typing import List, Callable
from functools import partial

import numpy as np

import torch

from split_llm.common.communication import Node
from split_llm.protocols.base import Protocol
from split_llm.protocols.base_protocols import SS_Mul__CX_N0_Y_N1, SS_Mul__CX_N0, SS_Perm
from split_llm.protocols.ss_mul_with_memory import SS_Mul__AppendingX
from split_llm.protocols.element_wise import SS_ElementWise__RandPerm
from split_llm.glm6b.wrapped_layer import Attention_GLM_Wrapped, GLMPositionalEmbedding
from split_llm.glm6b.utils import generate_position_ids

from split_llm.common.torch_utils import permute_2d_with_seed



def setup_node(node: Node, attention_layers: List[Attention_GLM_Wrapped]):
    node.space.attentions = attention_layers


class GLMAttentionProtocol(Protocol):
    def __init__(self, n0: Node, n1: Node, n2: Node, layer: int, mask_scale: float, max_generation_length: int = 500, device: str = "cpu"):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.layer = layer
    

        self.max_generation_length = max_generation_length
        self.mask_scale = mask_scale

        self.prompt_length = None
        self.total_length = None
        self.current_length = None
        self.position_ids = []
        self.positional_embedding = GLMPositionalEmbedding(64).to(device)

        self.device = device

        self.name = f"Attn_Layer_{layer}"

        self.qkv_mul_name = f"Attn_{layer}/qkv_matmul"
        self.dot_product_name = f"Attn_{layer}/dot_product"
        self.softmax_name = f"Attn_{layer}/softmax"
        self.weighted_sum_name = f"Attn_{layer}/weighted_sum"
        self.attn_out_name = f"Attn_{layer}/attn_out"

        self.qkv_mul_protocol = SS_Mul__CX_N0(
            [4096, 4096 * 3], 
            (lambda w, x: x @ w), self.qkv_mul_name,
            n0, n1, n2,
            mask_scale, device)
        
        self.dot_product_protocol = SS_Mul__AppendingX(
            [self.max_generation_length, 1, 32, 128], 0, 
            (lambda k, q: self.n0.space.attentions[self.layer].generate_logit_scores(q, k)),
            self.dot_product_name, 
            n0, n1, n2,
            mask_scale, device
        )

        self.softmax_protocol = SS_ElementWise__RandPerm(
            permute_2d_with_seed, partial(permute_2d_with_seed, reverse=True),
            partial(torch.softmax, dim=-1), self.softmax_name,
            n0, n1, n2,
            mask_scale, device
        )

        self.weighted_sum_protocol = SS_Mul__AppendingX(
            [self.max_generation_length, 1, 32, 128], 0,
            (lambda v, score: self.n0.space.attentions[self.layer].generate_weighted_values(score, v)),
            self.weighted_sum_name,
            n0, n1, n2,
            mask_scale, device
        )

        self.attn_out_protocol = SS_Mul__CX_N0(
            [4096, 4096], (lambda w, x: x @ w), self.attn_out_name,
            n0, n1, n2,
            mask_scale, device
        )

        self.n0.storage[f"{self.qkv_mul_name}:x"] = self.n0.space.attentions[self.layer].qkv_weight.T
        self.n0.storage[f"{self.attn_out_name}:x"] = self.n0.space.attentions[self.layer].attn_out_weight.T

    def prepare(self):
        self.qkv_mul_protocol.prepare()
        self.dot_product_protocol.prepare()
        self.softmax_protocol.prepare()
        self.weighted_sum_protocol.prepare()

    def offline_execute(self, next_length: int):
        if len(self.position_ids) == 0:
            self.prompt_length = next_length
            self.total_length = next_length
            self.position_ids.insert(0, generate_position_ids(self.prompt_length, self.total_length).to(self.device))
        else:
            self.total_length += next_length
            self.position_ids.insert(0, generate_position_ids(self.prompt_length, self.total_length).to(self.device))
        self.current_length = next_length

        self.qkv_mul_protocol.offline_execute([next_length, 1, 4096], [next_length, 1, 4096 * 3])
        self.dot_product_protocol.offline_execute([next_length, 1, 32, 128], [next_length, self.total_length, 1, 32], next_length)

        perm_key = np.random.randint(2 ** 30)
        self.n0.storage[f"{self.softmax_name}:new_perm"] = perm_key
        self.n0.storage[f"{self.softmax_name}:new_invperm"] = perm_key
        self.softmax_protocol.offline_execute([next_length * 32 * 1, self.total_length])
        self.weighted_sum_protocol.offline_execute([next_length, self.total_length, 1, 32], [next_length, 1, 4096], next_length)

    def online_step_qkv(self):
        """
        Input: [q_len, 1, 4096]
            Node 0: x0
            Node 1: x1

        Output: [q_len, 1, 4096 * 3]
            Node 0: h0
            Node 1: h1
        """
        self.n0.storage[f"{self.qkv_mul_name}:y0"] = self.n0.storage[f"{self.name}:x0"]
        self.n1.storage[f"{self.qkv_mul_name}:y1"] = self.n1.storage[f"{self.name}:x1"]

        self.qkv_mul_protocol.online_execute()

        self.n0.storage[f"{self.name}:h0"] = self.n0.storage[f"{self.qkv_mul_name}:z0"] + self.n0.space.attentions[self.layer].qkv_bias
        self.n1.storage[f"{self.name}:h1"] = self.n1.storage[f"{self.qkv_mul_name}:z1"]

        self.qkv_mul_protocol.clear_io()
    
    def online_step_dot_product(self):
        """
        Input: [q_len, 1, 4096 * 3]
            Node 0: h0
            Node 1: h1

        Output: [q_len, k_len, 1, 32]
            Node 0: s0
            Node 1: s1
        """
        position_ids = self.position_ids.pop()
        
        # In node_0
        self.n0.storage[f"{self.name}:q0"], self.n0.storage[f"{self.name}:k0"], self.n0.storage[f"{self.name}:v0"] = \
            self.n0.storage[f"{self.name}:h0"].view(-1, 1, 32, 128 * 3).chunk(3, dim=-1)

        self.n0.storage[f"{self.name}:q0"], self.n0.storage[f"{self.name}:k0"] = self.positional_embedding(
            self.n0.storage[f"{self.name}:q0"], self.n0.storage[f"{self.name}:k0"], position_ids
        )

        self.n0.storage[f"{self.dot_product_name}:x0 appended"] = self.n0.storage[f"{self.name}:k0"]
        self.n0.storage[f"{self.dot_product_name}:y0"] = self.n0.storage[f"{self.name}:q0"]

        # In node_1
        self.n1.storage[f"{self.name}:q1"], self.n1.storage[f"{self.name}:k1"], self.n1.storage[f"{self.name}:v1"] = \
            self.n1.storage[f"{self.name}:h1"].view(-1, 1, 32, 128 * 3).chunk(3, dim=-1)
        
        self.n1.storage[f"{self.name}:q1"], self.n1.storage[f"{self.name}:k1"] = self.positional_embedding(
            self.n1.storage[f"{self.name}:q1"], self.n1.storage[f"{self.name}:k1"], position_ids
        )

        self.n1.storage[f"{self.dot_product_name}:x1 appended"] = self.n1.storage[f"{self.name}:k1"]
        self.n1.storage[f"{self.dot_product_name}:y1"] = self.n1.storage[f"{self.name}:q1"]


        self.dot_product_protocol.online_execute()


        self.n0.storage[f"{self.name}:s0"] = self.n0.storage[f"{self.dot_product_name}:z0"]
        self.n1.storage[f"{self.name}:s1"] = self.n1.storage[f"{self.dot_product_name}:z1"]

        self.dot_product_protocol.clear_io()


    def online_step_softmax(self):
        """
        Input: [q_len, k_len, 1, 32]
            Node 0: s0
            Node 1: s1
        Output: [q_len, k_len, 1, 32]
            Node 0: s0
            Node 1: s1
        """
        # In node_0
        self.n0.storage[f"{self.softmax_name}:x0"] = self.n0.storage[f"{self.name}:s0"].swapaxes(1, 3).reshape(-1, self.total_length)
        # In node_1
        self.n1.storage[f"{self.softmax_name}:x1"] = self.n1.storage[f"{self.name}:s1"].swapaxes(1, 3).reshape(-1, self.total_length)

        self.softmax_protocol.online_execute()

        # In node_0
        self.n0.storage[f"{self.name}:s0"] = \
            self.n0.storage[f"{self.softmax_name}:z0"].view(self.current_length, 32, 1, self.total_length).swapaxes(1, 3)
        # In node_1
        self.n1.storage[f"{self.name}:s1"] = \
            self.n1.storage[f"{self.softmax_name}:z1"].view(self.current_length, 32, 1, self.total_length).swapaxes(1, 3)
        
        self.softmax_protocol.clear_io()


    def online_step_weighted_v(self):
        """
        Input: [q_len, k_len, 1, 32]
            Node 0: s0, v0
            Node 1: s1, v1
        Output: [q_len, k_len, 1, 32]
            Node 0: h0
            Node 1: h1
        """
        # In node_0
        self.n0.storage[f"{self.weighted_sum_name}:y0"] = self.n0.storage[f"{self.name}:s0"]
        self.n0.storage[f"{self.weighted_sum_name}:x0 appended"] = self.n0.storage[f"{self.name}:v0"]
        
        # In node_1
        self.n1.storage[f"{self.weighted_sum_name}:y1"] = self.n1.storage[f"{self.name}:s1"]
        self.n1.storage[f"{self.weighted_sum_name}:x1 appended"] = self.n1.storage[f"{self.name}:v1"]

        self.weighted_sum_protocol.online_execute()

        self.n0.storage[f"{self.name}:h0"] = self.n0.storage[f"{self.weighted_sum_name}:z0"]
        self.n1.storage[f"{self.name}:h1"] = self.n1.storage[f"{self.weighted_sum_name}:z1"]

    def online_step_attn_out(self):
        """
        Input: [q_len, 1, 4096]
            Node 0: h0
            Node 1: h1

        Output: [q_len, 1, 4096]
            Node 0: h0
            Node 1: h1
        """
        self.n0.storage[f"{self.attn_out_name}:y0"] = self.n0.storage[f"{self.name}:h0"]
        self.n1.storage[f"{self.attn_out_name}:y1"] = self.n1.storage[f"{self.name}:h1"]

        self.attn_out_protocol.online_execute()

        self.n0.storage[f"{self.name}:h0"] = self.n0.storage[f"{self.attn_out_name}:y0"]
        self.n1.storage[f"{self.name}:h1"] = self.n1.storage[f"{self.attn_out_name}:y1"]

    def online_execute(self):
        