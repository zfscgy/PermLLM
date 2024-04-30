from typing import Any, List, Dict, Callable, Union
from functools import partial

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


from split_llm.common.communication import Node
from split_llm.protocols.base import Protocol
from split_llm.protocols.base_protocols import SS_Mul__CX_N0_Y_N1, SS_Mul__CX_N0, SS_Perm
from split_llm.protocols.ss_mul_with_memory import SS_Mul__AppendingX
from split_llm.protocols.element_wise import SS_ElementWise__RandPerm
from split_llm.glm6b.wrapped_layer import Attention_GLM_Wrapped, GLMPositionalEmbedding, FeedForward_GLM_Wrapped
from split_llm.glm6b.utils import generate_position_ids

from split_llm.common.torch_utils import permute_2d_with_seed, permute_with_seed


from homomorphic_encryption.bfv import BFV
import tenseal as ts



class GLMConfig:
    model_dim = 4096
    n_heads = 32
    head_dim = 128
    n_tokens = 130006


def setup_node(node: Node, 
               attention_layers: List[Attention_GLM_Wrapped], 
               ff_layers: List[FeedForward_GLM_Wrapped],
               word_embedding: nn.Embedding,
               lm_head_layer: nn.Linear):
    node.space.attentions = attention_layers
    node.space.ffs = ff_layers
    node.space.word_embedding = word_embedding
    node.space.final_dense = lm_head_layer



def get_sub_dict(raw_dict: Dict[str, Any], prefix: str):
    sub_dict = dict()
    for k in raw_dict:
        if k.startswith(prefix):
            sub_dict[k.removeprefix(prefix)] = raw_dict[k]
    return sub_dict


class GLM_AttentionProtocol(Protocol):
    mask_scale_keys = ["qkv/u", "qkv/v", "qkv/w", 
                       "dot_product/u", "dot_product/v", "dot_product/w", 
                       "softmax/x", "softmax/z", 
                       "weighted_sum/u", "weighted_sum/v", "weighted_sum/w",
                       "attn_out/u", "attn_out/v", "attn_out/w"]
    def __init__(self, n0: Node, n1: Node, n2: Node, layer: int, mask_scale: Union[float, Dict[str, float]], max_generation_length: int = 500, name: str = None, device: str = "cpu"):
        """
        mask_scale keys (description): 
            qkv/u (h), qkv/v (qkv_weight), qkv/w
            dot_product/u (k), dot_product/v (q), dot_product/w
            softmax/x, softmax/z,
            weighted_sum/u, weighted_sum/v, weighted_sum/w,
            attn_out/u, attn_out/v, attn_out/w
        """
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.layer = layer
    

        self.max_generation_length = max_generation_length

        if not isinstance(mask_scale, dict):
            mask_scale = {k: mask_scale for k in self.mask_scale_keys}
        self.mask_scale = mask_scale

        self.prompt_length = None
        self.total_length = 0
        self.current_length = None
        self.position_ids = []
        self.positional_embedding = GLMPositionalEmbedding(64).to(device)

        self.device = device

        self.name = name or f"Attn_Layer_{layer}"

        self.qkv_mul_name = f"{self.name}/qkv_matmul"
        self.dot_product_name = f"{self.name}/dot_product"
        self.softmax_name = f"{self.name}/softmax"
        self.weighted_sum_name = f"{self.name}/weighted_sum"
        self.attn_out_name = f"{self.name}/attn_out"

        self.qkv_mul_protocol = SS_Mul__CX_N0(
            [GLMConfig.model_dim, GLMConfig.model_dim * 3], 
            (lambda w, x: x @ w), self.qkv_mul_name,
            n0, n1, n2,
            get_sub_dict(mask_scale, "qkv/"), device)
        
        # Find the local node
        local_node = None
        for node in [self.n0, self.n1, self.n2]:
            if node.local():
                local_node = node
                break

        self.dot_product_protocol = SS_Mul__AppendingX(
            [self.max_generation_length, 1, GLMConfig.n_heads, GLMConfig.head_dim], 0, 
            (lambda k, q: local_node.space.attentions[layer].generate_logit_scores(q, k)),
            self.dot_product_name, 
            n0, n1, n2,
            get_sub_dict(mask_scale, "dot_product/"), device
        )

        self.softmax_protocol = SS_ElementWise__RandPerm(
            permute_2d_with_seed, partial(permute_2d_with_seed, reverse=True),
            partial(torch.softmax, dim=-1), self.softmax_name,
            n0, n1, n2,
            get_sub_dict(mask_scale, "softmax/"), device
        )

        self.weighted_sum_protocol = SS_Mul__AppendingX(
            [self.max_generation_length, 1, GLMConfig.n_heads, GLMConfig.head_dim], 0,
            (lambda v, score: local_node.space.attentions[self.layer].generate_weighted_values(score, v)),
            self.weighted_sum_name,
            n0, n1, n2,
            get_sub_dict(mask_scale, "weighted_sum/"), device
        )

        self.attn_out_protocol = SS_Mul__CX_N0(
            [GLMConfig.model_dim, GLMConfig.model_dim], (lambda w, x: x @ w), self.attn_out_name,
            n0, n1, n2,
            get_sub_dict(mask_scale, "attn_out/"), device
        )

    def prepare(self):
        if self.n0.local():
            self.n0.storage[f"{self.qkv_mul_name}:x"] = self.n0.space.attentions[self.layer].qkv_weight.T
            self.n0.storage[f"{self.attn_out_name}:x"] = self.n0.space.attentions[self.layer].attn_out_weight.T


        self.qkv_mul_protocol.prepare()
        self.dot_product_protocol.prepare()
        self.softmax_protocol.prepare()
        self.weighted_sum_protocol.prepare()
        self.attn_out_protocol.prepare()


    def offline_execute(self, next_length: int):
        if self.total_length == 0:
            self.prompt_length = next_length
            self.total_length = next_length
            self.position_ids.insert(0, generate_position_ids(self.prompt_length, self.total_length)[..., -next_length:].to(self.device))
        else:
            self.total_length += next_length
            self.position_ids.insert(0, generate_position_ids(self.prompt_length, self.total_length)[..., -next_length:].to(self.device))
        self.current_length = next_length

        self.qkv_mul_protocol.offline_execute([next_length, 1, GLMConfig.model_dim], [next_length, 1, GLMConfig.model_dim * 3])
        self.dot_product_protocol.offline_execute([next_length, 1, GLMConfig.n_heads, GLMConfig.head_dim], [next_length, self.total_length, 1, GLMConfig.n_heads], next_length)

        if self.n0.local():
            perm_key = np.random.randint(2 ** 30)
            self.n0.storage[f"{self.softmax_name}:new_perm"] = perm_key
            self.n0.storage[f"{self.softmax_name}:new_invperm"] = perm_key

        self.softmax_protocol.offline_execute([next_length * GLMConfig.n_heads * 1, self.total_length])
        self.weighted_sum_protocol.offline_execute([next_length, self.total_length, 1, GLMConfig.n_heads], [next_length, 1, GLMConfig.model_dim], next_length)
        self.attn_out_protocol.offline_execute([next_length, 1, GLMConfig.model_dim], [next_length, 1, GLMConfig.model_dim])

    def online_step_qkv(self):
        """
        Input: [q_len, 1, model_dim]
            Node 0: x0
            Node 1: x1

        Output: [q_len, 1, model_dim * 3]
            Node 0: h0
            Node 1: h1
        """
        if self.n0.local():
            self.n0.storage[f"{self.qkv_mul_name}:y0"] = self.n0.storage[f"{self.name}:x0"]
        
        if self.n1.local():
            self.n1.storage[f"{self.qkv_mul_name}:y1"] = self.n1.storage[f"{self.name}:x1"]

        self.qkv_mul_protocol.online_execute()

        if self.n0.local():
            self.n0.storage[f"{self.name}:h0"] = self.n0.storage[f"{self.qkv_mul_name}:z0"] + self.n0.space.attentions[self.layer].qkv_bias
        
        if self.n1.local():
            self.n1.storage[f"{self.name}:h1"] = self.n1.storage[f"{self.qkv_mul_name}:z1"]

        self.qkv_mul_protocol.clear_io()
    
    def online_step_dot_product(self):
        """
        Input: [q_len, 1, model_dim * 3]
            Node 0: h0
            Node 1: h1

        Output: [q_len, k_len, 1, n_heads]
            Node 0: s0
            Node 1: s1
        """
        position_ids = self.position_ids.pop()
        
        # In node_0
        if self.n0.local():
            self.n0.storage[f"{self.name}:q0"], self.n0.storage[f"{self.name}:k0"], self.n0.storage[f"{self.name}:v0"] = \
                self.n0.storage[f"{self.name}:h0"].view(-1, 1, GLMConfig.n_heads, GLMConfig.head_dim * 3).chunk(3, dim=-1)

            self.n0.storage[f"{self.name}:q0"], self.n0.storage[f"{self.name}:k0"] = self.positional_embedding(
                self.n0.storage[f"{self.name}:q0"], self.n0.storage[f"{self.name}:k0"], position_ids
            )

            self.n0.storage[f"{self.dot_product_name}:x0 appended"] = self.n0.storage[f"{self.name}:k0"]
            self.n0.storage[f"{self.dot_product_name}:y0"] = self.n0.storage[f"{self.name}:q0"]

        # In node_1
        if self.n1.local():
            self.n1.storage[f"{self.name}:q1"], self.n1.storage[f"{self.name}:k1"], self.n1.storage[f"{self.name}:v1"] = \
                self.n1.storage[f"{self.name}:h1"].view(-1, 1, GLMConfig.n_heads, GLMConfig.head_dim * 3).chunk(3, dim=-1)
            
            self.n1.storage[f"{self.name}:q1"], self.n1.storage[f"{self.name}:k1"] = self.positional_embedding(
                self.n1.storage[f"{self.name}:q1"], self.n1.storage[f"{self.name}:k1"], position_ids
            )

            self.n1.storage[f"{self.dot_product_name}:x1 appended"] = self.n1.storage[f"{self.name}:k1"]
            self.n1.storage[f"{self.dot_product_name}:y1"] = self.n1.storage[f"{self.name}:q1"]


        self.dot_product_protocol.online_execute()

        if self.n0.local():
            self.n0.storage[f"{self.name}:s0"] = self.n0.storage[f"{self.dot_product_name}:z0"]
        
        if self.n1.local():
            self.n1.storage[f"{self.name}:s1"] = self.n1.storage[f"{self.dot_product_name}:z1"]

        self.dot_product_protocol.clear_io()


    def online_step_softmax(self):
        """
        Input: [q_len, k_len, 1, n_heads]
            Node 0: s0
            Node 1: s1
        Output: [q_len, k_len, 1, n_heads]
            Node 0: s0
            Node 1: s1
        """
        # In node_0
        if self.n0.local():
            self.n0.storage[f"{self.softmax_name}:x0"] = self.n0.storage[f"{self.name}:s0"].swapaxes(1, 3).reshape(-1, self.total_length)
        # In node_1
        if self.n1.local():
            self.n1.storage[f"{self.softmax_name}:x1"] = self.n1.storage[f"{self.name}:s1"].swapaxes(1, 3).reshape(-1, self.total_length)

        self.softmax_protocol.online_execute()

        # In node_0
        if self.n0.local():
            self.n0.storage[f"{self.name}:s0"] = \
                self.n0.storage[f"{self.softmax_name}:z0"].view(self.current_length, GLMConfig.n_heads, 1, self.total_length).swapaxes(1, 3)
        # In node_1
        if self.n1.local():
            self.n1.storage[f"{self.name}:s1"] = \
                self.n1.storage[f"{self.softmax_name}:z1"].view(self.current_length, GLMConfig.n_heads, 1, self.total_length).swapaxes(1, 3)
        
        self.softmax_protocol.clear_io()


    def online_step_weighted_v(self):
        """
        Input: [q_len, k_len, 1, n_heads]
            Node 0: s0, v0
            Node 1: s1, v1
        Output: [q_len, k_len, 1, n_heads]
            Node 0: h0
            Node 1: h1
        """
        # In node_0
        if self.n0.local():
            self.n0.storage[f"{self.weighted_sum_name}:y0"] = self.n0.storage[f"{self.name}:s0"]
            self.n0.storage[f"{self.weighted_sum_name}:x0 appended"] = self.n0.storage[f"{self.name}:v0"]
        
        # In node_1
        if self.n1.local():
            self.n1.storage[f"{self.weighted_sum_name}:y1"] = self.n1.storage[f"{self.name}:s1"]
            self.n1.storage[f"{self.weighted_sum_name}:x1 appended"] = self.n1.storage[f"{self.name}:v1"]

        self.weighted_sum_protocol.online_execute()

        if self.n0.local():
            self.n0.storage[f"{self.name}:h0"] = self.n0.storage[f"{self.weighted_sum_name}:z0"]
        
        if self.n1.local():
            self.n1.storage[f"{self.name}:h1"] = self.n1.storage[f"{self.weighted_sum_name}:z1"]

        self.weighted_sum_protocol.clear_io()

    def online_step_attn_out(self):
        """
        Input: [q_len, 1, model_dim]
            Node 0: h0
            Node 1: h1

        Output: [q_len, 1, model_dim]
            Node 0: h0
            Node 1: h1
        """

        if self.n0.local():
            self.n0.storage[f"{self.attn_out_name}:y0"] = self.n0.storage[f"{self.name}:h0"]
        if self.n1.local():
            self.n1.storage[f"{self.attn_out_name}:y1"] = self.n1.storage[f"{self.name}:h1"]

        self.attn_out_protocol.online_execute()

        if self.n0.local():
            self.n0.storage[f"{self.name}:z0"] = self.n0.storage[f"{self.attn_out_name}:z0"] + self.n0.space.attentions[self.layer].attn_out_bias
        if self.n1.local():
            self.n1.storage[f"{self.name}:z1"] = self.n1.storage[f"{self.attn_out_name}:z1"]

        self.attn_out_protocol.clear_io()

    def online_execute(self):
        """
        Input: [q_len, 1, model_dim]
            Node 0: x0
            Node 1: x1

        Output: [q_len, 1, .model_dim]
            Node 0: z0
            Node 1: z1
        """

        self.online_step_qkv()
        self.online_step_dot_product()
        self.online_step_softmax()
        self.online_step_weighted_v()
        self.online_step_attn_out()

    def clear_io(self):
        if self.n0.local():
            del self.n0.storage[f"{self.name}:x0"]
            del self.n0.storage[f"{self.name}:q0"], self.n0.storage[f"{self.name}:k0"], self.n0.storage[f"{self.name}:v0"]
            del self.n0.storage[f"{self.name}:s0"], self.n0.storage[f"{self.name}:h0"]
            del self.n0.storage[f"{self.name}:z0"]

        if self.n1.local():
            del self.n1.storage[f"{self.name}:x1"]
            del self.n1.storage[f"{self.name}:q1"], self.n1.storage[f"{self.name}:k1"], self.n1.storage[f"{self.name}:v1"]
            del self.n1.storage[f"{self.name}:s1"], self.n1.storage[f"{self.name}:h1"]
            del self.n1.storage[f"{self.name}:z1"]



class GLM_FeedForwardProtocol_PlainWeights(Protocol):
    mask_scale_keys = [
        "layernorm_in/x", "layernorm_in/z", 
        "gelu/x", "gelu/z", 
        "layernorm_out/x", "layernorm_out/z" 
    ]
    def __init__(self, n0: Node, n1: Node, n2: Node, layer: int, mask_scale: Union[float, Dict[str, float]], max_generation_length: int = 500, name: str = None, device: str = "cpu"):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.layer = layer
    

        self.max_generation_length = max_generation_length

        if not isinstance(mask_scale, dict):
            mask_scale = {k: mask_scale for k in self.mask_scale_keys}
        self.mask_scale = mask_scale

        self.device = device

        self.name = name or f"FF_Layer_{layer}"


        self.layernorm_in_name = f"{self.name}/layernorm_in"
        self.gelu_name = f"{self.name}/gelu"
        self.layernorm_out_name = f"{self.name}/layernorm_out"

        self.layernorm_in_protocol = SS_ElementWise__RandPerm(
            permute_2d_with_seed, partial(permute_2d_with_seed, reverse=True),
            partial(F.layer_norm, normalized_shape=[GLMConfig.model_dim]), self.layernorm_in_name,
            n0, n1, n2,
            get_sub_dict(mask_scale, "layernorm_in/"), device
        )

        self.gelu_protocol = SS_ElementWise__RandPerm(
            permute_with_seed, partial(permute_with_seed, reverse=True),
            F.gelu, self.gelu_name,
            n0, n1, n2,
            get_sub_dict(mask_scale, "gelu/"), device
        )

        self.layernorm_out_protocol = SS_ElementWise__RandPerm(
            permute_2d_with_seed, partial(permute_2d_with_seed, reverse=True),
            partial(F.layer_norm, normalized_shape=[GLMConfig.model_dim]), self.layernorm_out_name,
            n0, n1, n2,
            get_sub_dict(mask_scale, "layernorm_out/"), device
        )

    def prepare(self):
        self.layernorm_in_protocol.prepare()
        self.gelu_protocol.prepare()
        self.layernorm_out_protocol.prepare()

    def offline_execute(self, next_length: int):
        if self.n0.local():
            perm_key = np.random.randint(2 ** 30)
            self.n0.storage[f"{self.layernorm_in_name}:new_perm"] = perm_key
            self.n0.storage[f"{self.layernorm_in_name}:new_invperm"] = perm_key
        self.layernorm_in_protocol.offline_execute([next_length, GLMConfig.model_dim])
        
        if self.n0.local():
            perm_key = np.random.randint(2 ** 30)
            self.n0.storage[f"{self.gelu_name}:new_perm"] = perm_key
            self.n0.storage[f"{self.gelu_name}:new_invperm"] = perm_key
        self.gelu_protocol.offline_execute([next_length, 4 * GLMConfig.model_dim])
            
        if self.n0.local():
            perm_key = np.random.randint(2 ** 30)
            self.n0.storage[f"{self.layernorm_out_name}:new_perm"] = perm_key
            self.n0.storage[f"{self.layernorm_out_name}:new_invperm"] = perm_key
        self.layernorm_out_protocol.offline_execute([next_length, GLMConfig.model_dim])


    def online_step_layernorm_in(self):
        """
        Input:
            Node 0: h0
            Node 1: h1
        Output:
            Node 0: h0
            Node 1: h1
        """
        if self.n0.local():
            self.n0.storage[f"{self.layernorm_in_name}:x0"] = self.n0.storage[f"{self.name}:h0"]
        
        if self.n1.local():
            self.n1.storage[f"{self.layernorm_in_name}:x1"] = self.n1.storage[f"{self.name}:h1"]

        self.layernorm_in_protocol.online_execute()

        if self.n0.local():
            self.n0.storage[f"{self.name}:h0"] = \
                self.n0.storage[f"{self.layernorm_in_name}:z0"] * self.n0.space.ffs[self.layer].layernorm_in.weight + \
                self.n0.space.ffs[self.layer].layernorm_in.bias

        if self.n1.local():
            self.n1.storage[f"{self.name}:h1"] = \
                self.n1.storage[f"{self.layernorm_in_name}:z1"] * self.n1.space.ffs[self.layer].layernorm_in.weight

        self.layernorm_in_protocol.clear_io()

    def online_step_mlp_in(self):
        """
        Input:
            Node 0: h0
            Node 1: h1
        Output:
            Node 0: h0
            Node 1: h1
        """
        if self.n0.local():
            self.n0.storage[f"{self.name}:h0"] = \
                self.n0.storage[f"{self.name}:h0"] @ self.n0.space.ffs[self.layer].mlp_dense_in.weight.T + \
                self.n0.space.ffs[self.layer].mlp_dense_in.bias

        if self.n1.local():
            self.n1.storage[f"{self.name}:h1"] = \
                self.n1.storage[f"{self.name}:h1"] @ self.n1.space.ffs[self.layer].mlp_dense_in.weight.T
        
    def online_step_gelu(self):
        """
        Input:
            Node 0: h0
            Node 1: h1
        Output:
            Node 0: h0
            Node 1: h1
        """
        if self.n0.local():
            self.n0.storage[f"{self.gelu_name}:x0"] = self.n0.storage[f"{self.name}:h0"]
        if self.n1.local():
            self.n1.storage[f"{self.gelu_name}:x1"] = self.n1.storage[f"{self.name}:h1"]

        self.gelu_protocol.online_execute()

        if self.n0.local():
            self.n0.storage[f"{self.name}:h0"] = self.n0.storage[f"{self.gelu_name}:z0"]
        if self.n1.local():
            self.n1.storage[f"{self.name}:h1"] = self.n1.storage[f"{self.gelu_name}:z1"]
    
    def online_step_mlp_out(self):
        """
        Input:
            Node 0: h0
            Node 1: h1
        Output:
            Node 0: h0
            Node 1: h1
        """

        if self.n0.local():
            self.n0.storage[f"{self.name}:h0"] = \
                self.n0.storage[f"{self.name}:h0"] @ self.n0.space.ffs[self.layer].mlp_dense_out.weight.T + \
                self.n0.space.ffs[self.layer].mlp_dense_out.bias

        if self.n1.local():
            self.n1.storage[f"{self.name}:h1"] = \
                self.n1.storage[f"{self.name}:h1"] @ self.n1.space.ffs[self.layer].mlp_dense_out.weight.T
        
    def online_step_layernorm_out(self):
        """
        Input:
            Node 0: h0
            Node 1: h1
        Output:
            Node 0: h0
            Node 1: h1
        """

        if self.n0.local():
            self.n0.storage[f"{self.layernorm_out_name}:x0"] = self.n0.storage[f"{self.name}:h0"]
        if self.n1.local():
            self.n1.storage[f"{self.layernorm_out_name}:x1"] = self.n1.storage[f"{self.name}:h1"]

        self.layernorm_out_protocol.online_execute()

        
        if self.n0.local() and not isinstance(self.n0.space.ffs[self.layer].layernorm_out, nn.Identity):
            self.n0.storage[f"{self.name}:h0"] = \
                self.n0.storage[f"{self.layernorm_out_name}:z0"] * self.n0.space.ffs[self.layer].layernorm_out.weight + \
                self.n0.space.ffs[self.layer].layernorm_out.bias

        if self.n1.local() and not isinstance(self.n1.space.ffs[self.layer].layernorm_out, nn.Identity):
            self.n1.storage[f"{self.name}:h1"] = \
                self.n1.storage[f"{self.layernorm_out_name}:z1"] * self.n1.space.ffs[self.layer].layernorm_out.weight

        self.layernorm_out_protocol.clear_io()

    def online_execute(self):
        """
        Input:
            Node 0: x0
            Node 1: x1
        Output:
            Node 0: z0
            Node 1: z1
        """
        
        if self.n0.local():
            self.n0.storage[f"{self.name}:h0"] = self.n0.storage[f"{self.name}:x0"].view(-1, GLMConfig.model_dim)
        
        if self.n1.local():
            self.n1.storage[f"{self.name}:h1"] = self.n1.storage[f"{self.name}:x1"].view(-1, GLMConfig.model_dim)

        self.online_step_layernorm_in()

        if self.n0.local():
            self.n0.storage[f"{self.name}:h_in_0"] = self.n0.storage[f"{self.name}:h0"]
        
        if self.n1.local():
            self.n1.storage[f"{self.name}:h_in_1"] = self.n1.storage[f"{self.name}:h1"]


        self.online_step_mlp_in()
        self.online_step_gelu()
        self.online_step_mlp_out()

        if self.n0.local():
            self.n0.storage[f"{self.name}:h0"] += (2 * 28) ** 0.5 * self.n0.storage[f"{self.name}:h_in_0"] 
        
        if self.n1.local():
            self.n1.storage[f"{self.name}:h1"] += (2 * 28) ** 0.5 * self.n1.storage[f"{self.name}:h_in_1"]

        self.online_step_layernorm_out()

        if self.n0.local():
            self.n0.storage[f"{self.name}:z0"] = self.n0.storage[f"{self.name}:h0"].view(-1, 1, GLMConfig.model_dim)

        if self.n1.local():
            self.n1.storage[f"{self.name}:z1"] = self.n1.storage[f"{self.name}:h1"].view(-1, 1, GLMConfig.model_dim)

    def clear_io(self):
        if self.n0.local():
            del self.n0.storage[f"{self.name}:x0"]
            del self.n0.storage[f"{self.name}:z0"]

        if self.n1.local():
            del self.n1.storage[f"{self.name}:x1"]
            del self.n1.storage[f"{self.name}:z1"]


class GLM_TransformerLayerProtocol(Protocol):
    mask_scale_keys = GLM_AttentionProtocol.mask_scale_keys + GLM_FeedForwardProtocol_PlainWeights.mask_scale_keys
    def __init__(self, n0: Node, n1: Node, n2: Node, layer: int, mask_scale: float, max_generation_length: int = 500, name: str = None, device: str = "cpu"):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.layer = layer
    

        self.max_generation_length = max_generation_length
        self.mask_scale = mask_scale

        self.device = device

        self.name = name or f"GLM__Transformer_{layer}"

        self.current_length = None

        self.attn_name = f"{self.name}/attn"
        self.ff_name = f"{self.name}/ff"

        self.attn_protocol = GLM_AttentionProtocol(n0, n1, n2, layer, mask_scale, max_generation_length, self.attn_name, device)
        self.ff_protocol = GLM_FeedForwardProtocol_PlainWeights(n0, n1, n2, layer, mask_scale, max_generation_length, self.ff_name, device)

    def prepare(self):
        self.attn_protocol.prepare()
        self.ff_protocol.prepare()

    def offline_execute(self, next_length: int):
        self.attn_protocol.offline_execute(next_length)
        self.ff_protocol.offline_execute(next_length)
    
    def online_execute(self):
        """
        Input:
            Node 0: x0
            Node 1: x1
        Output:
            Node 0: z0
            Node 1: z1
        """
        if self.n0.local():
            self.n0.storage[f"{self.attn_name}:x0"] = self.n0.storage[f"{self.name}:x0"]
        
        if self.n1.local():
            self.n1.storage[f"{self.attn_name}:x1"] = self.n1.storage[f"{self.name}:x1"]
        
        self.attn_protocol.online_execute()

        if self.n0.local():
            self.n0.storage[f"{self.ff_name}:x0"] = self.n0.storage[f"{self.attn_name}:x0"] + (2 * 28) ** 0.5 * self.n0.storage[f"{self.name}:x0"]
        if self.n1.local():
            self.n1.storage[f"{self.ff_name}:x1"] = self.n1.storage[f"{self.attn_name}:x1"] + (2 * 28) ** 0.5 * self.n1.storage[f"{self.name}:x1"]
        

        self.ff_protocol.online_execute()

        if self.n0.local():
            self.n0.storage[f"{self.name}:z0"] = self.n0.storage[f"{self.ff_name}:z0"]
        
        if self.n1.local():
            self.n1.storage[f"{self.name}:z1"] = self.n1.storage[f"{self.ff_name}:z1"]
        
        self.attn_protocol.clear_io()
        self.ff_protocol.clear_io()

    def clear_io(self):
        if self.n0.local():
            del self.n0.storage[f"{self.name}:x0"]
            del self.n0.storage[f"{self.name}:z0"]

        if self.n1.local():
            del self.n1.storage[f"{self.name}:x1"]
            del self.n1.storage[f"{self.name}:z1"]


class GLM_PredictionProtocol(Protocol):
    mask_scale_keys = ["final_dense/u","final_dense/v", "final_dense/w", "score_permutation"]
    def __init__(self, n0: Node, n1: Node, n2: Node, mask_scale: Union[float, Dict[str, float]], name: str = None, device: str = "cpu"):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2

        if not isinstance(mask_scale, dict):
            mask_scale = {k: mask_scale for k in self.mask_scale_keys}
        self.mask_scale = mask_scale

        self.device = device

        self.name = name or f"GLM__PredictionLayer"

        self.current_length = None

        self.prediction_dense_name = f"{self.name}/final_dense"
        self.randperm_name = f"{self.name}/score_permutation"

        self.prediction_dense_protocol = SS_Mul__CX_N0(
            [GLMConfig.model_dim, GLMConfig.n_tokens], (lambda x, y: y @ x), self.prediction_dense_name,
            n0, n1, n2, get_sub_dict(mask_scale, "final_dense/"), device
        )

        self.randperm_protocol = SS_Perm(
            (lambda x, i: x[i]), self.randperm_name, n0, n1, n2, mask_scale["score_permutation"], device
        )

    def prepare(self):
        if self.n0.local():
            last_dense: nn.Linear = self.n0.space.final_dense
            self.n0.storage[f"{self.prediction_dense_name}:x"] = last_dense.weight[:GLMConfig.n_tokens].T

        self.prediction_dense_protocol.prepare()
        self.randperm_protocol.prepare()

        # In node_1
        if self.n1.local():
            self.n1.space.bfv_cryptosystem = BFV()
            self.n1.send(self.n0.name, f"{self.name}:bfv_keys", self.n1.space.bfv_cryptosystem.serialize())

        # In node_0
        if self.n0.local():
            self.n0.space.bfv_cryptosystem = BFV.from_bytes(self.n0.fetch(self.n1.name, f"{self.name}:bfv_keys"))

    def offline_execute(self):
        self.prediction_dense_protocol.offline_execute([GLMConfig.model_dim], [GLMConfig.n_tokens])
        
        if self.n0.local():
            perm_key = torch.randperm(GLMConfig.n_tokens, device=self.device)
            self.n0.storage[f"{self.randperm_name}:new_perm"] = perm_key
        self.randperm_protocol.offline_execute([GLMConfig.n_tokens])
    
    def online_execute(self):
        """
        Input:
            Node 0: y0
            Node 1: y1
        """
        if self.n0.local():
            self.n0.storage[f"{self.prediction_dense_name}:y0"] = self.n0.storage[f"{self.name}:x0"]
        
        if self.n1.local():
            self.n1.storage[f"{self.prediction_dense_name}:y1"] = self.n1.storage[f"{self.name}:x1"]
        
        self.prediction_dense_protocol.online_execute()

        if self.n0.local():
            self.n0.storage[f"{self.name}:current_permutation"] = self.n0.storage[f"{self.randperm_name}:perm"][-1].tolist()
            self.n0.storage[f"{self.randperm_name}:x0"] = self.n0.storage[f"{self.prediction_dense_name}:z0"]  # The final prediction layer has no bias
        
        if self.n1.local():
            self.n1.storage[f"{self.randperm_name}:x1"] = self.n1.storage[f"{self.prediction_dense_name}:z1"]
        
        self.randperm_protocol.online_execute()

        self.n0.send(self.n1.name, f"{self.randperm_name}:permuted_s0", self.n0.storage[f"{self.randperm_name}:z0"])

        # In node_1
        if self.n1.local():
            permuted_scores = self.n1.storage[f"{self.randperm_name}:z1"] + \
                self.n1.fetch(self.n0.name, f"{self.randperm_name}:permuted_s0")

            # Using the greedy generation
            best_idx = np.argmax(permuted_scores.tolist())
            indicator = np.zeros([GLMConfig.n_tokens], dtype=int)
            indicator[best_idx] = 1
            step_size = self.n1.space.bfv_cryptosystem.ciphertext_size
            indicator_cts = []
            for i in range(0, GLMConfig.n_tokens, step_size):
                indicator_cts.append(self.n1.space.bfv_cryptosystem.enc_vector(indicator[i: i + step_size]))
            self.n1.send(self.n0.name, f"{self.name}:index_indicator_ct", [c.serialize() for c in indicator_cts])
            del permuted_scores, best_idx, indicator, step_size, indicator_cts

        # In node_0
        if self.n0.local():
            indicator_ct_bytes: List[bytes] = self.n0.fetch(self.n1.name, f"{self.name}:index_indicator_ct")
            indicator_cts = [ts.bfv_vector_from(self.n0.space.bfv_cryptosystem.context, b) for b in indicator_ct_bytes]
            index_cts = []
            step_size = self.n1.space.bfv_cryptosystem.ciphertext_size
            for i in range(0, GLMConfig.n_tokens, step_size):
                index_cts.append(indicator_cts[i // step_size].dot(self.n0.storage[f"{self.name}:current_permutation"] [i:i + step_size]))        
            index_ct: ts.BFVVector = sum(index_cts[1:], start=index_cts[0])
        
            self.n0.send(self.n1.name, f"{self.name}:index__ct", index_ct.serialize())

            del indicator_ct_bytes, indicator_cts, index_cts, index_ct

        # In node_1
        if self.n1.local():
            index_ct = ts.bfv_vector_from(self.n1.space.bfv_cryptosystem.context, self.n1.fetch(self.n0.name, f"{self.name}:index__ct"))
            index = self.n1.space.bfv_cryptosystem.decrypt(index_ct)
            self.n1.storage[f"{self.name}:z"] = index

            del index_ct, index
    
        self.prediction_dense_protocol.clear_io()
        self.randperm_protocol.clear_io()


    def clear_io(self):
        if self.n0.local():
            del self.n0.storage[f"{self.name}:x0"]
        
        if self.n1.local():
            del self.n1.storage[f"{self.name}:x1"], self.n1.storage[f"{self.name}:z"]


class GLM_EmbeddingRetrievalProtocol(Protocol):
    def __init__(self, n0: Node, n1: Node, n2: Node, mask_scale: float, name: str = None, device: str = "cpu"):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
    

        self.mask_scale = mask_scale

        self.device = device

        self.name = name or f"GLM__EmbeddingLayer"

        self.current_length = None

        self.embedding_retrieval_name = f"{self.name}/embedding_retrieval"

        self.embedding_retrieval_protocol = SS_Mul__CX_N0_Y_N1(
            [GLMConfig.n_tokens, GLMConfig.model_dim], (lambda x, y: y @ x), self.embedding_retrieval_name,
            n0, n1, n2, mask_scale, device
        )

    def prepare(self):
        if self.n0.local():
            embedding: nn.Linear = self.n0.space.word_embedding
            self.n0.storage[f"{self.embedding_retrieval_name}:x"] = embedding.weight
        self.embedding_retrieval_protocol.prepare()

    def offline_execute(self, next_length: int):
        self.embedding_retrieval_protocol.offline_execute([next_length, GLMConfig.n_tokens], [next_length, GLMConfig.model_dim])

    def online_execute(self):
        """
        Input:
            Node 1: x
        """
        if self.n1.local():
            self.n1.storage[f"{self.embedding_retrieval_name}:y"] = self.n1.storage[f"{self.name}:x"]
    
        self.embedding_retrieval_protocol.online_execute()
        
        if self.n0.local():
            self.n0.storage[f"{self.name}:z0"] = self.n0.storage[f"{self.embedding_retrieval_name}:z0"]
        if self.n1.local():
            self.n1.storage[f"{self.name}:z1"] = self.n1.storage[f"{self.embedding_retrieval_name}:z1"]

    def clear_io(self):
        if self.n0.local():
            del self.n0.storage[f"{self.name}:z0"]

        if self.n1.local():
            del self.n1.storage[f"{self.name}:x"]
            del self.n1.storage[f"{self.name}:z1"]


class GLM_Protocol(Protocol):
    def __init__(self, n0: Node, n1: Node, n2: Node, mask_scale: float, max_generation_length: int, name: str = None, device: str = "cpu"):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
    

        # Convert mask_scale into dict
        if not isinstance(mask_scale, dict):
            mask_scale_dict: dict = dict()
            mask_scale_dict.update({"embedding_retrieval/u": mask_scale, "embedding_retrieval/v": mask_scale, "embedding_retrieval/w": mask_scale})
            mask_scale_dict.update({"prediction/" + k: mask_scale for k in GLM_PredictionProtocol.mask_scale_keys})
            for layer in range(28):  # there are total 28 layers in GLM
                mask_scale_dict.update({f"transformer_layer_{layer}/" + k: mask_scale for k in GLM_TransformerLayerProtocol.mask_scale_keys})

        mask_scale = mask_scale_dict

        self.mask_scale = mask_scale

        self.device = device

        self.name = name or f"GLM__Whole"

        self.embedding_retrieval_name = "embedding_retrieval"
        self.embedding_retrieval_protocol = GLM_EmbeddingRetrievalProtocol(
            n0, n1, n2, get_sub_dict(mask_scale, "embedding_retrieval/"), self.embedding_retrieval_name, device
        )

        self.layer_names = [f"transformer_layer_{i}" for i in range(28)]
        self.layer_protocols = [
            GLM_TransformerLayerProtocol(n0, n1, n2, i, get_sub_dict(mask_scale, f"transformer_layer_{i}/"), max_generation_length, self.layer_names[i], device)
            for i in range(28)
        ]
        
        self.prediction_name = "prediction"
        self.prediction_protocol = GLM_PredictionProtocol(n0, n1, n2, get_sub_dict(mask_scale, "prediction/"), self.prediction_name, device)


    def prepare(self):
        self.embedding_retrieval_protocol.prepare()
        for layer_protocol in self.layer_protocols:
            layer_protocol.prepare()
        self.prediction_protocol.prepare()

    
    def offline_execute(self, next_length: int):
        self.embedding_retrieval_protocol.offline_execute(next_length)
        for layer_protocol in self.layer_protocols:
            layer_protocol.offline_execute(next_length)
        self.prediction_protocol.offline_execute()

    def online_execute(self):
        if self.n1.local():
            self.n1.storage[f"{self.embedding_retrieval_name}:x"] = self.n1.storage[f"{self.name}:x"]
        self.embedding_retrieval_protocol.online_execute()

        if self.n0.local():
            self.n0.storage[f"{self.layer_names[0]}:x0"] = self.n0.storage[f"{self.embedding_retrieval_name}:z0"][:, None]  
            # adding the batch dimension
        if self.n1.local():
            self.n1.storage[f"{self.layer_names[0]}:x1"] = self.n1.storage[f"{self.embedding_retrieval_name}:z1"][:, None]
        
        self.embedding_retrieval_protocol.clear_io()

        for i in range(28):
            self.layer_protocols[i].online_execute()
            if i != 27:  # Except the last layer
                if self.n0.local():
                    self.n0.storage[f"{self.layer_names[i + 1]}:x0"] = self.n0.storage[f"{self.layer_names[i]}:z0"]
                if self.n1.local():
                    self.n1.storage[f"{self.layer_names[i + 1]}:x1"] = self.n1.storage[f"{self.layer_names[i]}:z1"]

                self.layer_protocols[i].clear_io()


        i = 27  # Here is the last transformer layer
        if self.n0.local():
            self.n0.storage[f"{self.prediction_name}:x0"] = self.n0.storage[f"{self.layer_names[i]}:z0"][-1, 0]  # Extract the last embedding
        if self.n1.local():
            self.n1.storage[f"{self.prediction_name}:x1"] = self.n1.storage[f"{self.layer_names[i]}:z1"][-1, 0]

        self.layer_protocols[i].clear_io()

        self.prediction_protocol.online_execute()

        self.n1.storage[f"{self.name}:z"] = self.n1.storage[f"{self.prediction_name}:z"]

        self.prediction_protocol.clear_io()

    def clear_io(self):
        if self.n1.local():
            del self.n1.storage[f"{self.name}:x"], self.n1.storage[f"{self.name}:z"]