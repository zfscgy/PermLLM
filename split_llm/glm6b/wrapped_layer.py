from turtle import position
from typing import List, Tuple, Union
from dataclasses import dataclass

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from llm_bases.chatglm6b_official.modeling_chatglm import GLMBlock, RotaryEmbedding, SelfAttention, attention_fn
from split_llm.common.utils import random_orthonormal, inverse_permutation, quantize, random_vec_with_seed
from split_llm.glm6b.configs import GLM6BConfig
from split_llm.glm6b.utils import get_rotary_embedding, rotate_half


class GLMPositionalEmbedding_Raw(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        dim: dimension of the each head's embedding
        """
        self.inv_freq = nn.Parameter(1. / (10000 ** (torch.arange(0, dim, 2).half() / dim)), requires_grad=False)
        # [dim]

    def get_rotary_embedding(self, seq_len: int):
        v = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)[:, None] @ self.inv_freq[None, :]
        v = torch.cat([v, v], dim=-1)
        return torch.cos(v), torch.sin(v)  # [seq_len, dim / 2], [seq_len, dim / 2]

    def forward(self, qs: torch.Tensor, ks: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        qs, ks: [seq_len, batch, n_heads, head_dim]
        position_ids: [batch, 2, seq_len]
        """
        cos_emb, sin_emb = self.get_rotary_embedding(torch.max(position_ids))  # [seq_len, dim/2]
        qs1, qs2 = qs.chunk(2, dim=-1)
        ks1, ks2 = ks.chunk(2, dim=-1)
        position_ids_1 = position_ids[:, 0, :].T  # [seq_len, batch]
        position_ids_2 = position_ids[:, 1, :].T  # [seq_len, batch]

        def apply_position_embedding(xs: torch.Tensor, position_ids: torch.Tensor):
            """
            xs: [seq_len, batch, dim/2]
            position_ids: [seq_len, batch]
            """
            cos_embs = F.embedding(position_ids, cos_emb)  # [seq_len, batch, dim/2]
            sin_embs = F.embedding(position_ids, sin_emb)
            xs = (xs * cos_embs) + (rotate_half(xs) * sin_embs)
            return xs
        
        qs1 = apply_position_embedding(qs1, position_ids_1)
        qs2 = apply_position_embedding(qs2, position_ids_2)
        ks1 = apply_position_embedding(ks1, position_ids_1)
        ks2 = apply_position_embedding(ks2, position_ids_2)
        qs = torch.concat([qs1, qs2], dim=-1)
        ks = torch.concat([ks1, ks2], dim=-1)
        
        return qs, ks


def attention_fn_raw(
        q, k, v,
        attention_mask,
        model_dim,
        layer_id,
        scaling_attention_score=True
):
    """
    q, k, v: [seq_len, batch, n_heads, head_dim]
    attention_mask: []
    Raw implementation of the attention in GLM6B
    """
    # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
    seq_len, b, nh, hidden_size = k.shape
    query_key_layer_scaling_coeff = float(layer_id + 1)
    if scaling_attention_score:
        q = q / (np.sqrt(hidden_size) * query_key_layer_scaling_coeff)
    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================
    # [b, np, sq, sk]
    output_size = (q.size(1), q.size(2), q.size(0), k.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    q = q.view(output_size[2], output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    k = k.view(output_size[3], output_size[0] * output_size[1], -1)

    matmul_result = torch.zeros(
        1, 1, 1,
        dtype=q.dtype,
        device=q.device,
    )

    matmul_result = torch.baddbmm(
        matmul_result,
        q.transpose(0, 1),  # [b * np, sq, hn]
        k.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        beta=0.0,
        alpha=1.0,
    )

    # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    if not (attention_mask == 0).all():
        # if auto-regressive, skip
        attention_scores.masked_fill_(attention_mask, -10000.0)
    dtype = attention_scores.dtype
    attention_scores = attention_scores.half()
    attention_scores = attention_scores * query_key_layer_scaling_coeff

    attention_probs = F.softmax(attention_scores, dim=-1)

    attention_probs = attention_probs.type(dtype)

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]
    output_size = (v.size(1), v.size(2), q.size(0), v.size(3))

    # change view [sk, b * np, hn]
    v = v.view(v.size(0), output_size[0] * output_size[1], -1)

    # change view [b * np, sq, sk]
    attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

    # matmul: [b * np, sq, hn]
    context = torch.bmm(attention_probs, v.transpose(0, 1))

    # change view [b, np, sq, hn]
    context = context.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context = context.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context.size()[:-2] + (model_dim,)
    context = context.view(*new_context_layer_shape)

    return context


class Attention(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, layer_id: int):
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.layer_id = layer_id

        self.qkv_weight = nn.Parameter(torch.zeros(model_dim, 3 * model_dim, dtype=torch.float))
        self.qkv_bias = nn.Parameter(torch.zeros(3 * model_dim, dtype=torch.float))
        
        self.attn_out_weight = nn.Parameter(torch.zeros(model_dim, model_dim, type=torch.float))
        self.attn_out_bias = nn.Parameter(torch.zeros(model_dim, model_dim, dtype=torch.float))

        self.positional_embedding = GLMPositionalEmbedding_Raw(model_dim // n_heads)

    def generate_logit_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        q: [query_len, batch, n_heads, head_dim]
        k: [key_len, batch, n_heads, head_dim]
        k can contain different key vectors, so the first dimension could be different than q
        """
        q = q[:, None]  # [q_len, 1, batch, n_heads, head_dim]
        k = k[None, :]  # [1, k_len, batch, n_heads, head_dim]

        logits = torch.sum(q * k, dim=-1)  # [q_len, k_len, batch, n_heads]
        return logits

    def generate_softmax_scores(self, logit_scores: torch.Tensor, dim: int=1) -> torch.Tensor:
        """
        It seems that attention_mask is useless during the inference!
        """
        return F.softmax(logit_scores)
    
    def generate_weighted_values(self, softmax_scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        softmax_scores: [q_len, k_len, batch, n_heads]
        v: [k_len, batch, n_heads, head_dim]
        """
        q_len, k_len, batch, n_heads = softmax_scores.shape
        softmax_scores = softmax_scores[:, :, :, :, None]
        v = v[None, :, :, :, :]
        weighted_v = torch.sum(softmax_scores * v, dim=1)  # [q_len, batch, n_heads, head_sim]
        weighted_v = weighted_v.view(q_len, batch, -1)  # [q_len, batch, model_dim]
        return weighted_v

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        q, k, v = (x @ self.qkv_weight + self.qkv_bias).chunk(3, dim=-1)
        q, k = self.positional_embedding(q, k, position_ids)
        logit_scores = self.generate_logit_scores(q, k)
        softmax_scores = self.generate_softmax_scores(logit_scores)
        weighted_v = self.generate_weighted_values(softmax_scores, v)

        attn_out = (weighted_v @ self.attn_out_bias + self.attn_out_bias)
        return attn_out


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, layer_id: int):
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
        h3 = self.mlp_dense_out(h2)
        h4 = h3 + self.residual_coef * h1
        h5 = self.layernorm_out(h4)
        return h5


def copy_attantion():
    pass


def copy_feedforward():
    pass



class GLMBlockInputTransform(nn.Module):
    def __init__(self, original_glm: GLMBlock):
        super(GLMBlockInputTransform, self).__init__()
        self.model_dim = original_glm.attention.hidden_size
        self.n_attention_heads = original_glm.attention.num_attention_heads

        self.input_affine = nn.Linear(self.model_dim, self.model_dim)
        self.linear_qkv = nn.Linear(self.model_dim, self.model_dim * 3)

        self.input_affine.weight.data = torch.diag(original_glm.input_layernorm.weight.data)
        self.input_affine.bias.data = original_glm.input_layernorm.bias.data.clone()



        self.linear_qkv.weight.data = original_glm.attention.query_key_value.weight.data.clone()
        self.linear_qkv.bias.data = original_glm.attention.query_key_value.bias.data.clone()

        dim = self.model_dim // (self.n_attention_heads * 2)
        # [head_dim / 2]
        # [head_dim / 4]


    def get_rotary_embedding(self, seq_length: torch.LongTensor):
        v = torch.arange(seq_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)[:, None] @ self.inv_freq[None, :]
        v = torch.cat([v, v], dim=-1)
        return torch.cos(v), torch.sin(v)  # [seq_len, head_dim / 2], [seq_len, head_dim / 2]

    def apply_position_embedding(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        rotary_embedding = self.get_rotary_embedding(torch.max(position_ids) + 1)

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions

        def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
            # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
            cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
                       F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
            q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
            return q, k

        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        cos, sin = rotary_embedding
        position_ids, block_position_ids = position_ids[:, 0, :].transpose(0, 1).contiguous(), \
                                           position_ids[:, 1, :].transpose(0, 1).contiguous()
        q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
        q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
        q = torch.concat([q1, q2], dim=-1)
        k = torch.concat([k1, k2], dim=-1)
        return q, k


    def projection_part_transform(self, xs: torch.Tensor, position_ids: torch.Tensor):
        qkv = xs @ self.input_affine.weight.T @ self.linear_qkv.weight.T
        qkv = qkv.view(*qkv.shape[:-1], self.n_attention_heads, 3 * self.model_dim // self.n_attention_heads)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.apply_position_embedding(q, k, position_ids)
        return q, k, v

    def translation_part_transform(self, position_ids: torch.Tensor):
        qkv = self.input_affine.bias @ self.linear_qkv.weight.T + self.linear_qkv.bias
        qkv = qkv.view(*qkv.shape[:-1], self.n_attention_heads, 3 * self.model_dim // self.n_attention_heads)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.apply_position_embedding(q, k, position_ids)
        return q, k, v
    
    def forward(self, xs: torch.Tensor, position_ids: torch.Tensor=None, only_projection: bool=False, only_affine: bool=False):
        if only_affine:
            y = xs @ self.input_affine.weight.T
            if not only_projection:
                y += self.input_affine.bias
            return y
        else:
            q0, k0, v0 = self.projection_part_transform(xs, position_ids)
            if not only_projection:
                q1, k1, v1 = self.translation_part_transform(position_ids)
                return q0 + q1, k0 + k1, v0 + v1
            else:
                return q0, k0, v0
        


@dataclass
class ObfuscationKeys:
    """
    This is the obfuscation key for one single transformer layer

    qkv: Tuple of two tensors:
        * Tensor of [n_heads, head_dim, head_dim] for transforamtion of each head
        * Tensor of [n_heads] for the permutation index

    attn_out: Tensor of [model_dim, model_dim]
    mlp_hidden: Tensor of [mlp_hidden_dim] for the permutation idnex
    mlp_output

    """
    qkv: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    attn_out: torch.Tensor
    mlp_hidden: torch.Tensor
    mlp_output: torch.Tensor


def keys_to_tensor(keys: ObfuscationKeys, device: str="cpu", float_type: torch.dtype=torch.float, int_type: torch.dtype=torch.int):
    def to_float_tensor(x):
        # For integer tensors, this is not needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x.type(float_type).to(device)

    def to_int_tensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x.type(int_type).to(device)
    
    return ObfuscationKeys(
        qkv=((to_float_tensor(keys.qkv[0][0]), to_int_tensor(keys.qkv[0][1])),
             (to_float_tensor(keys.qkv[1][0]), to_int_tensor(keys.qkv[1][1])),
             (to_float_tensor(keys.qkv[2][0]), to_int_tensor(keys.qkv[2][1]))),
        attn_out=to_float_tensor(keys.attn_out),
        mlp_hidden=to_int_tensor(keys.mlp_hidden),
        mlp_output=to_float_tensor(keys.mlp_output)
    )



def generate_obfuscation_keys(model_dim: int, n_heads: int, device:str="cpu") -> ObfuscationKeys:
    def get_multihead_keys():
        return torch.stack([random_orthonormal(model_dim // n_heads, device=device) for _ in range(n_heads)]), \
               torch.tensor(np.random.permutation(n_heads), device=device)
    trans_qk, perm_qk = get_multihead_keys()
    trans_v, _ = get_multihead_keys()
    return keys_to_tensor(ObfuscationKeys(
        qkv=((trans_qk, perm_qk), (trans_qk, perm_qk), (trans_v, perm_qk)),
        attn_out=random_orthonormal(model_dim),
        mlp_hidden=torch.tensor(np.random.permutation(4 * model_dim), device=device),  # the hidden dim in MLP is 4 time the model dim
        mlp_output=random_orthonormal(model_dim)
    ))


def expand_segmented_keys(trans: torch.Tensor, perm: torch.Tensor):
    n_segs = trans.shape[0]
    seg_dim = trans.shape[1]
    combiend_key = torch.zeros([n_segs * seg_dim, n_segs * seg_dim], dtype=trans.dtype, device=trans.device)
    for i in range(n_segs):
        combiend_key[perm[i] * seg_dim: (perm[i] + 1) * seg_dim, perm[i] * seg_dim: (perm[i] + 1) * seg_dim] = trans[i]
    return combiend_key


def obfuscate_transformer(source_transformer: WrappedGLMBlock, keys: ObfuscationKeys, device: str="cuda"):
    """
    Obfuscate the transformer in device, and returns in CPU
    """
    transformer = WrappedGLMBlock(source_transformer.layer_id)
    copy_module(source_transformer, transformer)
    transformer = transformer.float().to(device)
    keys = keys_to_tensor(keys, device)
    attn_cxt_key = expand_segmented_keys(*keys.qkv[2])

    # Obfuscate the attention output
    transformer.attn_linear.weight.data = keys.attn_out @ transformer.attn_linear.weight.data @ attn_cxt_key.T
    transformer.attn_linear.bias.data = keys.attn_out @ transformer.attn_linear.bias.data

    # Obfuscate the affine weights
    # Due to the residual connection, the keys is the same as mlp out key
    # * Notice: the bias are masked to increase the attacker difficulty
    mlp_input_key = keys.mlp_output

    transformer.post_attention_affine.weight.data = mlp_input_key @ transformer.post_attention_affine.weight.data @ keys.attn_out.T
    affine_bias_mask = torch.zeros_like(transformer.post_attention_affine.bias.data)
    transformer.post_attention_affine.bias.data = mlp_input_key @ transformer.post_attention_affine.bias.data - affine_bias_mask
    
    # Obfuscate the mlp hidden layer
    transformer.mlp_linear1.weight.data = (transformer.mlp_linear1.weight.data @ mlp_input_key.T)[keys.mlp_hidden]
    transformer.mlp_linear1.bias.data = transformer.mlp_linear1.bias.data[keys.mlp_hidden] + transformer.mlp_linear1.weight.data @ affine_bias_mask

    # Obfuscate the mlp out layer
    transformer.mlp_linear2.weight.data = keys.mlp_output @ transformer.mlp_linear2.weight.data[:, keys.mlp_hidden]
    transformer.mlp_linear2.bias.data = keys.mlp_output @ transformer.mlp_linear2.bias.data + affine_bias_mask * (2 * 28) ** 0.5

    return transformer.cpu()


if __name__ == '__main__':
    def test_obfuscate_one_layer():
        from llm_bases.chatglm6b import ChatGML6B
        device_name = "cuda:2"

        glm = ChatGML6B()
        obf_layer = 10

        transformer_to_obfuscate = glm.condgen.transformer.layers[obf_layer].half().to(device_name)

        # Convert to the float
        input_transform = GLMBlockInputTransform(transformer_to_obfuscate)
        wrapped_glm = WrappedGLMBlock(transformer_to_obfuscate.layer_id)
        wrapped_glm.wrap(transformer_to_obfuscate)
        wrapped_glm.float().to(device_name)

        keys_0 = generate_obfuscation_keys(GLM6BConfig.model_dim, GLM6BConfig.n_attention_heads, device_name)
        keys_1 = generate_obfuscation_keys(GLM6BConfig.model_dim, GLM6BConfig.n_attention_heads, device_name)
        keys_0 = keys_to_tensor(keys_0, device=device_name, float_type=torch.float)
        keys_1 = keys_to_tensor(keys_1, device=device_name, float_type=torch.float)

        obfuscated_glm = obfuscate_transformer(obfuscate_transformer(wrapped_glm, keys_0, device_name), keys_1, device_name).half().to(device_name)
        
        keys_0 = keys_to_tensor(keys_0, device=device_name, float_type=torch.half)
        keys_1 = keys_to_tensor(keys_1, device=device_name, float_type=torch.half)
        
        input_transform = input_transform.half().to(device_name)
        wrapped_glm.half().to(device_name)

        glm.device = device_name
        glm.condgen.transformer.word_embeddings.to(device_name)
        glm.condgen.lm_head.to(device_name)



        tokenization = glm.get_tokenization("你是谁？")
        input_state = torch.tensor(random_vec_with_seed(1926, [5, 1, GLM6BConfig.model_dim], [-1, 1])).half().to(device_name)
        expected_output0 = transformer_to_obfuscate(input_state, tokenization[1], tokenization[2], torch.tensor(obf_layer))[0]
        expected_output0 = F.layer_norm(expected_output0, [4096])
        input_state = F.layer_norm(input_state, [4096])


        q, k, v = input_transform(input_state, tokenization[1])
        residual = input_transform(input_state, only_affine=True)

        expand_q_key_0 = expand_segmented_keys(*keys_0.qkv[0])
        expand_q_key_1 = expand_segmented_keys(*keys_1.qkv[0])

        expand_k_key_0 = expand_segmented_keys(*keys_0.qkv[1])
        expand_k_key_1 = expand_segmented_keys(*keys_1.qkv[1])

        expand_v_key_0 = expand_segmented_keys(*keys_0.qkv[2])
        expand_v_key_1 = expand_segmented_keys(*keys_1.qkv[2])

        qo = q.view(*q.shape[:-2], GLM6BConfig.model_dim) @ expand_q_key_0.T @ expand_q_key_1.T
        qo = qo.view(*q.shape[:-2], GLM6BConfig.n_attention_heads, GLM6BConfig.model_dim // GLM6BConfig.n_attention_heads)

        ko = k.view(*k.shape[:-2], GLM6BConfig.model_dim) @ expand_k_key_0.T @ expand_k_key_1.T
        ko = ko.view(*k.shape[:-2], GLM6BConfig.n_attention_heads, GLM6BConfig.model_dim // GLM6BConfig.n_attention_heads)

        vo = v.view(*v.shape[:-2], GLM6BConfig.model_dim) @ expand_v_key_0.T @ expand_v_key_1.T
        vo = vo.view(*v.shape[:-2], GLM6BConfig.n_attention_heads, GLM6BConfig.model_dim // GLM6BConfig.n_attention_heads)

        expected_output1 = wrapped_glm((q, k, v), tokenization[2].to(device_name), residual)

        obfuscated_output = obfuscated_glm((qo, ko, vo), tokenization[2].to(device_name), residual @ keys_0.attn_out.T @ keys_1.attn_out.T)
        actual_output = obfuscated_output @ keys_1.mlp_output @ keys_0.mlp_output

        print(expected_output0)
        print(expected_output1)
        print(actual_output)


    # test_wrapped_glm_block()
    test_obfuscate_one_layer()
