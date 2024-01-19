from typing import List, Tuple, Union
from dataclasses import dataclass

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from llm_bases.chatglm6b_official.modeling_chatglm import GLMBlock, SelfAttention, attention_fn
from desi_llm.common.utils import random_orthogonal, inverse_permutation, quantize
from desi_llm.glm6b.configs import GLM6BConfig



class WrappedGLMBlock(nn.Module):
    def __init__(self, layer_id: int):
        super(WrappedGLMBlock, self).__init__()
        self.model_dim = GLM6BConfig.model_dim
        self.layer_id = layer_id
        self.num_attention_heads = GLM6BConfig.n_attention_heads

        self.attn_linear = nn.Linear(self.model_dim, self.model_dim, bias=True)
        self.post_attention_affine = nn.Linear(self.model_dim, self.model_dim, bias=True)
        self.mlp_linear1 = nn.Linear(self.model_dim, self.model_dim * 4)
        self.mlp_linear2 = nn.Linear(self.model_dim * 4, self.model_dim)

    def wrap(self, original_glm_block: GLMBlock):
        # Clone all the weights
        self.attn_linear.weight.data = original_glm_block.attention.dense.weight.data.clone()
        self.attn_linear.bias.data = original_glm_block.attention.dense.bias.data.clone()

        self.post_attention_affine.weight.data = torch.diag(quantize(original_glm_block.post_attention_layernorm.weight.data, GLM6BConfig.n_attention_heads))
        self.post_attention_affine.bias.data = original_glm_block.post_attention_layernorm.bias.data.clone()

        self.mlp_linear1.weight.data = original_glm_block.mlp.dense_h_to_4h.weight.data.clone()
        self.mlp_linear1.bias.data = original_glm_block.mlp.dense_h_to_4h.bias.data.clone()
        self.mlp_linear2.weight.data = original_glm_block.mlp.dense_4h_to_h.weight.data.clone()
        self.mlp_linear2.bias.data = original_glm_block.mlp.dense_4h_to_h.bias.data.clone()

    def attention_fn(
            self,
            q, k, v,
            attention_mask,
            model_dim,
            layer_id,
            scaling_attention_score=True
    ):
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
        attention_scores = attention_scores.float()
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

    def forward(self, qkv: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor,
                attn_out_residual: torch.Tensor):
        """
        All the params are received from a previous node.
        :param qkv:
        :param attention_mask:
        :param attn_out_residual:
        :return:
        """
        q, k, v = qkv
        attention_context = self.attention_fn(q, k, v, attention_mask, self.model_dim, layer_id=self.layer_id)
        # [seq_len, batch, hidden_size]

        residual_coef = (2 * 28) ** 0.5

        attention_output = self.attn_linear(attention_context) + residual_coef * attn_out_residual
        attention_output_normalized = F.layer_norm(attention_output, [self.model_dim])
        affine_output = self.post_attention_affine(attention_output_normalized)
        # [seq_len, batch, hidden_size]

        mlp_hidden = self.mlp_linear1(affine_output)
        mlp_hidden_activated = F.gelu(mlp_hidden)
        mlp_output = self.mlp_linear2(mlp_hidden_activated) + residual_coef * affine_output
        # [seq_len, batch, hidden_size]

        normalized_output = F.layer_norm(mlp_output, [self.model_dim])

        return normalized_output


def copy_module(source_module: nn.Module, target_module: nn.Module):
    for p_source, p_target in zip(source_module.parameters(), target_module.parameters()):
        p_target.data = p_source.data.clone()


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

        self.inv_freq = nn.Parameter(1. / (10000 ** (torch.arange(0, dim, 2).float() / dim)), requires_grad=False)
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
        return q, k, v


TensorType = Union[torch.Tensor, np.ndarray]

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
    qkv: Tuple[Tuple[TensorType, TensorType], Tuple[TensorType, TensorType], Tuple[TensorType, TensorType]]
    attn_out: TensorType
    mlp_hidden: TensorType
    mlp_output: TensorType


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



def generate_obfuscation_keys(model_dim: int, n_heads: int) -> ObfuscationKeys:
    def get_multihead_keys():
        return np.stack([random_orthogonal(model_dim // n_heads) for _ in range(n_heads)]), np.random.permutation(n_heads)

    return keys_to_tensor(ObfuscationKeys(
        qkv=(get_multihead_keys(), get_multihead_keys(), get_multihead_keys()),
        attn_out=random_orthogonal(model_dim),
        mlp_hidden=np.random.permutation(4 * model_dim),  # the hidden dim in MLP is 4 time the model dim
        mlp_output=random_orthogonal(model_dim)
    ))


def expand_segmented_keys(trans: TensorType, perm: TensorType):
    n_segs = trans.shape[0]
    seg_dim = trans.shape[1]
    if isinstance(trans, np.ndarray):
        combiend_key = np.zeros([n_segs * seg_dim, n_segs * seg_dim], dtype=np.float32)
    else:
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
    transformer.attn_linear.bias.data = keys.attn_out @ transformer.attn_linear.bias.data @ attn_cxt_key.T

    # Obfuscate the affine weights
    # Due to the residual connection, the keys is the same as mlp out key
    # * Notice: the bias are masked to increase the attacker difficulty
    mlp_input_key = keys.mlp_output

    transformer.post_attention_affine.weight.data = mlp_input_key @ transformer.post_attention_affine.weight.data @ keys.attn_out.T
    affine_bias_mask = torch.rand_like(transformer.post_attention_affine.bias.data)
    transformer.post_attention_affine.bias.data = mlp_input_key @ transformer.post_attention_affine.bias.data - affine_bias_mask
    
    # Obfuscate the mlp hidden layer
    transformer.mlp_linear1.weight.data = (transformer.mlp_linear1.weight.data @ mlp_input_key.T)[keys.mlp_hidden]
    transformer.mlp_linear1.bias.data = (transformer.mlp_linear1.bias.data - transformer.mlp_linear1.weight @ affine_bias_mask)[keys.mlp_hidden]

    # Obfuscate the mlp out layer
    transformer.mlp_linear2.weight.data = (keys.mlp_output @ transformer.mlp_linear2.weight.data)[:, keys.mlp_hidden]
    transformer.mlp_linear2.bias.data = keys.mlp_output @ transformer.mlp_linear2.bias.data + affine_bias_mask * (2 * 28) ** 0.5

    return transformer.to("cpu")


if __name__ == '__main__':
    def test_obfuscate_one_layer():
        from llm_bases.chatglm6b import ChatGML6B
        glm = ChatGML6B()
        obf_layer = 10

        transformer_to_obfuscate = glm.condgen.transformer.layers[obf_layer].float()

        # Convert to the float
        input_transform = GLMBlockInputTransform(transformer_to_obfuscate)
        wrapped_glm = WrappedGLMBlock(transformer_to_obfuscate.layer_id)
        wrapped_glm.wrap(transformer_to_obfuscate)

        keys = generate_obfuscation_keys(GLM6BConfig.model_dim, GLM6BConfig.n_attention_heads)
        expand_v_key = expand_segmented_keys(*keys.qkv[2])
        obfuscate_transformer(wrapped_glm, keys)

        device_name = "cuda:1"
        glm.device = device_name
        glm.condgen.to(device_name)
        wrapped_glm = wrapped_glm.half().to(device_name)
        input_transform = input_transform.half().to(device_name)
        expand_v_key = torch.tensor(expand_v_key).half().to(device_name)
        keys = keys_to_tensor(keys, dtype=torch.half, device=device_name)

        def probability_generate_with_obfuscated(
                input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor):
            hidden_state = glm.get_initial_state(input_ids)
            hidden_state = glm.forward_layers(hidden_state, position_ids, attention_mask, 0, obf_layer)

            # Test with the original transformer
            # hidden_state = glm.condgen.transformer.layers[obf_layer](hidden_state, position_ids, attention_mask, obf_layer - 1)[0]

            q0, k0, v0 = input_transform.projection_part_transform(hidden_state, position_ids)
            q1, k1, v1 = input_transform.translation_part_transform(position_ids)
            q, k, v = q0 + q1, k0 + k1, v0 + v1

            v = v.view(*v.shape[:-2], GLM6BConfig.model_dim) @ expand_v_key.T
            v = v.view(*v.shape[:-1], GLM6BConfig.n_attention_heads, GLM6BConfig.model_dim // GLM6BConfig.n_attention_heads)

            hidden_state = wrapped_glm((q, k, v), attention_mask, hidden_state @ expand_v_key.T)

            hidden_state = hidden_state @ keys.mlp_output

            final_emb = glm.forward_layers(hidden_state, position_ids, attention_mask, obf_layer + 1)

            logits = glm.condgen.lm_head(final_emb).permute(1, 0, 2).contiguous()[..., -1, :]
            # Get the logits on the next position

            probs = torch.softmax(logits, dim=-1)  # [batch, n_tokens]
            return probs

        resp = glm.greedy_generate("你是谁？", partial(probability_generate_with_obfuscated))
        print(resp)

    # test_wrapped_glm_block()
    test_obfuscate_one_layer()
