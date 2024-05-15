from turtle import position
from typing import List, Tuple, Union
from dataclasses import dataclass

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from llm_bases.chatglm6b_official.modeling_chatglm import GLMBlock
from perm_llm.common.torch_utils import copy_param
from perm_llm.glm6b.utils import rotate_half, gelu_openai


class GLMPositionalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        dim: dimension of the each head's embedding
        """
        super(GLMPositionalEmbedding, self).__init__()
        self.inv_freq = nn.Parameter(1. / (10000 ** (torch.arange(0, dim, 2).float() / dim)), requires_grad=False)
        # [dim]
        coss, sins = self.generate_rotary_embedding(64)
        self.sin_embs = nn.Parameter(sins, requires_grad=False)
        self.cos_embs = nn.Parameter(coss, requires_grad=False)

    def generate_rotary_embedding(self, seq_len: int):
        v = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)[:, None] @ self.inv_freq[None, :]
        v = torch.cat([v, v], dim=-1)
        return torch.cos(v), torch.sin(v)  # [seq_len, dim / 2], [seq_len, dim / 2]

    def forward(self, qs: torch.Tensor, ks: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        qs, ks: [seq_len, batch, n_heads, head_dim]
        position_ids: [batch, 2, seq_len]
        """
        current_rotary_embedding_size = self.sin_embs.shape[0]
        max_len = torch.max(position_ids) + 1
        if max_len > current_rotary_embedding_size:
            self.cos_embs.data, self.sin_embs.data = self.generate_rotary_embedding(max_len * 2)

        qs1, qs2 = qs.chunk(2, dim=-1)
        ks1, ks2 = ks.chunk(2, dim=-1)
        position_ids_1 = position_ids[:, 0, :].T  # [seq_len, batch]
        position_ids_2 = position_ids[:, 1, :].T  # [seq_len, batch]

        def apply_position_embedding(xs: torch.Tensor, position_ids: torch.Tensor):
            """
            xs: [seq_len, batch, n_heads, head_dim/2]
            position_ids: [seq_len, batch]
            """
            cos_embs = F.embedding(position_ids, self.cos_embs)  # [seq_len, batch, head_dim/2]
            sin_embs = F.embedding(position_ids, self.sin_embs)
            xs = (xs * cos_embs[:, :, None, :]) + (rotate_half(xs) * sin_embs[:, :, None, :])
            return xs
        
        qs1 = apply_position_embedding(qs1, position_ids_1)
        qs2 = apply_position_embedding(qs2, position_ids_2)
        ks1 = apply_position_embedding(ks1, position_ids_1)
        ks2 = apply_position_embedding(ks2, position_ids_2)
        qs = torch.concat([qs1, qs2], dim=-1)
        ks = torch.concat([ks1, ks2], dim=-1)
        
        return qs, ks


class Attention_GLM_Wrapped(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, layer_id: int):
        super(Attention_GLM_Wrapped, self).__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.layer_id = layer_id

        self.qkv_weight = nn.Parameter(torch.zeros(3 * model_dim, model_dim, dtype=torch.float))
        self.qkv_bias = nn.Parameter(torch.zeros(3 * model_dim, dtype=torch.float))
        
        self.attn_out_weight = nn.Parameter(torch.zeros(model_dim, model_dim, dtype=torch.float))
        self.attn_out_bias = nn.Parameter(torch.zeros(model_dim, dtype=torch.float))

        self.positional_embedding = GLMPositionalEmbedding(model_dim // (2 * n_heads))

    def generate_qkv(self, x: torch.Tensor, position_ids: torch.Tensor):
        qkv = (x @ self.qkv_weight.T + self.qkv_bias).view(*x.shape[:2], self.n_heads, 3 * self.model_dim // self.n_heads)
        # Notice the order here: first divide into multiple heads, then each head is split into q, k, v

        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.positional_embedding(q, k, position_ids)
        
        return q, k, v

    def generate_logit_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        q: [query_len, batch, n_heads, head_dim]
        k: [key_len, batch, n_heads, head_dim]
        k can contain different key vectors, so the first dimension could be different than q
        """
        q = q

        q = q[:, None]  # [q_len, 1, batch, n_heads, head_dim]
        k = k[None, :]  # [1, k_len, batch, n_heads, head_dim]

        logits = torch.sum(q * k, dim=-1)  / np.sqrt(self.model_dim // self.n_heads) # [q_len, k_len, batch, n_heads]
        return logits

    def generate_softmax_scores(self, logit_scores: torch.Tensor, dim: int=1) -> torch.Tensor:
        """
        logit_scores: [q_len, k_len, batch, n_heads]
        It seems that attention_mask is useless during the inference!
        """
        return F.softmax(logit_scores, dim)  # [q_len, k_len, batch, n_heads]
    
    def generate_weighted_values(self, softmax_scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        softmax_scores: [q_len, k_len, batch, n_heads]
        v: [k_len, batch, n_heads, head_dim]
        """
        q_len, k_len, batch, n_heads = softmax_scores.shape
        softmax_scores = softmax_scores[:, :, :, :, None]  # [q_len, k_len, batch, n_heads, 1       ]
        v = v[None, :, :, :, :]                           #  [1,     k_len, batch, n_heads, head_sim]
        weighted_v = torch.sum(softmax_scores * v, dim=1)  # [q_len, batch, n_heads, head_sim]
        weighted_v = weighted_v.reshape(q_len, batch, -1)  # [q_len, batch, model_dim]
        return weighted_v

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        q, k, v = self.generate_qkv(x, position_ids)
        logit_scores = self.generate_logit_scores(q, k)
        softmax_scores = self.generate_softmax_scores(logit_scores)  # [q_len, ]
        weighted_v = self.generate_weighted_values(softmax_scores, v)
        attn_out = weighted_v @ self.attn_out_weight.T + self.attn_out_bias
        return attn_out
    


class FeedForward_GLM_Wrapped(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, layer_id: int):
        super(FeedForward_GLM_Wrapped, self).__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.layer_id = layer_id

        self.layernorm_in = nn.LayerNorm([model_dim])
        self.mlp_dense_in = nn.Linear(model_dim, 4 * model_dim)
        self.mlp_dense_out = nn.Linear(4 * model_dim, model_dim)
        self.layernorm_out = nn.LayerNorm([model_dim])

        self.residual_coef = (2 * 28) ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.layernorm_in(x)
        h1 = self.mlp_dense_in(h0)
        h2 = F.gelu(h1)
        #  h2 = gelu_openai(h1)
        #  Those two gelu implementations do not have significant difference
        h3 = self.mlp_dense_out(h2)
        h4 = h3 + self.residual_coef * h0
        h5 = self.layernorm_out(h4)
        return h5


def copy_attention(glm_block: GLMBlock, attn_layer: Attention_GLM_Wrapped):
    copy_param(glm_block.attention.query_key_value.weight, attn_layer.qkv_weight)
    copy_param(glm_block.attention.query_key_value.bias, attn_layer.qkv_bias)
    copy_param(glm_block.attention.dense.weight, attn_layer.attn_out_weight)
    copy_param(glm_block.attention.dense.bias, attn_layer.attn_out_bias)


def copy_feedforward(glm_block: GLMBlock, next_glm_block: GLMBlock, feed_forward: FeedForward_GLM_Wrapped):
    copy_param(glm_block.post_attention_layernorm.weight, feed_forward.layernorm_in.weight)
    copy_param(glm_block.post_attention_layernorm.bias, feed_forward.layernorm_in.bias)
    copy_param(glm_block.mlp.dense_h_to_4h.weight, feed_forward.mlp_dense_in.weight)
    copy_param(glm_block.mlp.dense_h_to_4h.bias, feed_forward.mlp_dense_in.bias)
    copy_param(glm_block.mlp.dense_4h_to_h.weight, feed_forward.mlp_dense_out.weight)
    copy_param(glm_block.mlp.dense_4h_to_h.bias, feed_forward.mlp_dense_out.bias)
    if next_glm_block is not None:
        copy_param(next_glm_block.input_layernorm.weight, feed_forward.layernorm_out.weight)
        copy_param(next_glm_block.input_layernorm.bias, feed_forward.layernorm_out.bias)
    else:
        feed_forward.layernorm_out = None  # No layernorm at the last transformer!



def share_attention(attn_raw: Attention_GLM_Wrapped, uniform_scale: float):
    attn_share_0 = Attention_GLM_Wrapped(attn_raw.model_dim, attn_raw.n_heads, attn_raw.layer_id)
    attn_share_1 = Attention_GLM_Wrapped(attn_raw.model_dim, attn_raw.n_heads, attn_raw.layer_id)
    
    qkv_w_0 = torch.rand_like(attn_raw.qkv_weight) * uniform_scale - uniform_scale / 2
    qkv_b_0 = torch.rand_like(attn_raw.qkv_bias) * uniform_scale - uniform_scale / 2

    dense_w_0 = torch.rand_like(attn_raw.attn_out_weight) * uniform_scale - uniform_scale / 2
    dense_b_0 = torch.rand_like(attn_raw.attn_out_bias) * uniform_scale - uniform_scale / 2

    attn_share_0.qkv_weight = qkv_w_0
    attn_share_0.qkv_bias = qkv_b_0
    attn_share_0.attn_out_weight.data = dense_w_0
    attn_share_0.attn_out_bias.data = dense_b_0

    
    attn_share_1.qkv_weight = attn_share_0.qkv_weight - qkv_w_0
    attn_share_1.qkv_bias = attn_share_0.qkv_bias - qkv_b_0
    attn_share_1.attn_out_weight.data = attn_share_0.attn_out_weight.data - dense_w_0
    attn_share_1.attn_out_bias.data = attn_share_0.attn_out_bias.data - dense_b_0

    return attn_share_0, attn_share_1

