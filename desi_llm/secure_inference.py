from logging import config
from typing import List
from dataclasses import dataclass


import torch
torch.set_grad_enabled(False)
import numpy as np


import tqdm
from zmq import device


from llm_bases.chatglm6b import ChatGML6B
from simple_pir.pir import PIRClient
from ckks.crypto import CKKS

from desi_llm.glm6b.obfuscated_layer import WrappedGLMBlock, keys_to_tensor
from desi_llm.glm6b.configs import GLM6BConfig
from desi_llm.glm6b.utils import generate_position_ids, generate_attention_mask

from desi_llm.nodes.computation_node import ComputationNode
from desi_llm.nodes.model_provider import ModelProvider
from desi_llm.nodes.obfuscator import ObfuscatorNode

from desi_llm.common.utils import random_vec_with_seed, generate_random_transformations, generate_random_linear_combination, reconstruct_random_linear_combination



def estimate_size(m):
    if isinstance(m, np.ndarray):
        if m.dtype == np.float32:
            return m.size * 4
        elif m.dtype in [np.int32, np.int64, np.uint32, np.uint64]:
            bit_length = np.ceil(np.log2(np.max(m) - np.min(m)))
            return m.size * bit_length / 4
        else:
            raise ValueError("Unsupported NumPy type for size estimation")
    if isinstance(m, torch.Tensor):
        if m.dtype == torch.float:
            return np.prod(m.shape) * 4
        elif m.dtype == torch.half:
            return np.prod(m.shape) * 2
        elif m.dtype in [torch.int, torch.long]:
            bit_length = torch.ceil(torch.log2(np.max(m) - torch.min(m))).item()
            return np.prod(m.shape) * bit_length / 4
        else:
            raise ValueError("Unsupported Torch type for size estimation")

    elif isinstance(m, list) or isinstance(m, tuple):
        return sum([estimate_size(mm) for mm in m])
    else:
        raise ValueError("Unsupported type for size estimation")

class NetworkSimulator:
    def __init__(self, latency: float, speed: float):
        """
        latency: time lag
        speed: number of **bytes** sent per minute
        """
        self.latency = latency
        self.speed = speed
    
        self.total_time: int = 0
        self.total_comm: float = 0

        self.all_history = []
    
    def transfer(self, m, desc: str = ""):
        m_size = estimate_size(m)
        self.total_comm += m_size

        m_time = self.latency + (self.total_comm) / self.speed
        self.total_time += m_time
        self.all_history.append((m_size, m_time, desc))


@dataclass
class DesiLLMConfig:
    device: str
    n_random_vectors_word_embedding: int
    noise_std_word_embedding: float
    n_random_vectors_hidden: int
    noise_std_vector_hidden: float
    n_random_vectors_final: int
    noise_std_vector_final: int


class DesiLLM:
    def __init__(self, network_simulator: NetworkSimulator, config: DesiLLMConfig):
        print("Load the original ChatGLM6B model to memory...")
        # Load model to the CPU memory
        self.glm6b = ChatGML6B()
        self.network_simulator = network_simulator
        self.config = config

        print("Initialize model provider...")
        # Model provider work:
        self.model_provider = ModelProvider(self.glm6b)

        self.obfuscators = []
        self.computation_nodes = []
        for layer in tqdm.tqdm(self.glm6b.condgen.transformer.layers):
            self.obfuscators.append(ObfuscatorNode())
            self.computation_nodes.append(ComputationNode(WrappedGLMBlock(layer.layer_id)))

        print("Model provider setup PIR sever...")
        # Generte PIR hints
        self.rotated_word_embedding_share_obfuscator = self.model_provider.generate_shared_embedding()
        pir_lwe_mat, pir_hint = self.model_provider.setup_pir_server()
        print("User creating PIR client...")
        self.pir_client = PIRClient(pir_lwe_mat, pir_hint, self.model_provider.pir_server.get_scale_factor(), self.model_provider.pir_server.plain_modulus)


        print("Start to load saved obfuscated models...")
        # Load all the obfuscated models
        model_save_dir = "./saved_model/"

        for i, computation_node in tqdm.tqdm(enumerate(self.computation_nodes)):
            computation_node.layer.load_state_dict(torch.load(model_save_dir + f"wrappedGLM_{i}.pth", map_location="cpu"))

        for i, obfuscator in tqdm.tqdm(enumerate(self.obfuscators)):
            obfuscator.key = torch.load(model_save_dir + f"obfuscatorKey_{i}.pth", map_location="cpu")
            self.model_provider.obfuscators[i].key = torch.load(model_save_dir + f"providerKey_{i}.pth", map_location="cpu")
        

        print("Start to load models to the designated device...")
        device = config.device
        for i in tqdm.tqdm(range(len(self.model_provider.input_transformations))):
            self.model_provider.input_transformations[i] = self.model_provider.input_transformations[i].half().to(device)
        for node in tqdm.tqdm(self.computation_nodes):
            node.layer = node.layer.half().to(device)
        for o in tqdm.tqdm(self.model_provider.obfuscators + self.obfuscators):
            o.key = keys_to_tensor(o.key, float_type=torch.half, int_type=torch.int)
            o.key.qkv = [[e1.to(device), e2.to(device)] for e1, e2 in o.key.qkv]
            o.key.mlp_output = o.key.mlp_output.to(device)
            o.key.attn_out=  o.key.attn_out.to(device)
        self.model_provider.word_embedding_key_device = torch.tensor(self.model_provider.word_embedding_key).half().to(device)
        self.glm6b.condgen.lm_head.to(device)

        self.ckks_scheme = CKKS()


    def retrieve_word_embedding(self, token_ids: List[int]):
        query_ids = token_ids + [t + self.glm6b.n_tokens for t in token_ids]
        # Client make PIR queries
        pir_queries = [self.pir_client.query(i) for i in query_ids]
        self.network_simulator.transfer(np.array(pir_queries), "pir: U->M")

        # PIR server response
        pir_answers = [self.model_provider.pir_server.answer(q) for q in pir_queries]
        self.network_simulator.transfer(np.array(pir_answers), "pir: M->U")

        recovered_ids = [self.pir_client.recover(i, a) for i, a in zip(query_ids, pir_answers)]
        recovered_ids = np.array(recovered_ids)
        permutation_ids = recovered_ids[:len(recovered_ids) // 2] * 1000 + recovered_ids[len(recovered_ids) // 2:]

        # User build embedding shares
        embedding_share_0 = []
        for perm_id in permutation_ids:
            embedding_share_0.append(random_vec_with_seed(perm_id, GLM6BConfig.model_dim, [-1, 1]))
        embedding_share_0 = np.array(embedding_share_0)
        # The user re-mask the word embedding
        embedding_share_0 += random_vec_with_seed(19260817, embedding_share_0.shape, [-1, 1])
        

        # User query the word embedding obfuscator
        self.network_simulator.transfer(np.array(permutation_ids), "permutationL U->O0")
        # The transformation can also be synced via a random seed
        embedding_share_1 = self.rotated_word_embedding_share_obfuscator[permutation_ids]
        embedding_share_1 -= random_vec_with_seed(19260817, embedding_share_1.shape, [-1, 1])

        embedding_share_0 = torch.tensor(embedding_share_0).half().to(self.config.device)
        embedding_share_1 = torch.tensor(embedding_share_1).half().to(self.config.device)

        return embedding_share_0, embedding_share_1


    def main_body_inference(self, embedding_share_0: torch.Tensor, embedding_share_1: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor):
        seq_len = embedding_share_0.shape[0]
        # User generate random linear combination for embedding_share_0
        embedding_share_0 = embedding_share_0[:, None, :]
        embedding_share_0 += torch.normal(0, 0.5, embedding_share_0.shape, dtype=embedding_share_0.dtype, device=self.config.device)
        random_transformation = generate_random_transformations(seq_len, self.config.n_random_vectors_word_embedding, ensure_sum_one=False)
        rlcs_0 = generate_random_linear_combination(embedding_share_0, random_transformation[0].half().to(self.config.device))
        
        rlcs_1 = embedding_share_1[:, None, :]

        # Both user and obfuscator send back to the model provider
        rlcs_0 = rlcs_0 @ self.model_provider.word_embedding_key_device
        rlcs_1 = rlcs_1 @ self.model_provider.word_embedding_key_device

        random_transformation = (random_transformation[0].half().to(self.config.device), random_transformation[1].half().to(self.config.device))

        # The model provider perform the computation at the beginning part of the transformer
        for i in tqdm.tqdm(range(len(self.obfuscators))):
            qkv_rlcs_proj = self.model_provider.input_transformations[i](rlcs_0, position_ids, only_projection=True)
            qkv_rlcs_tran = self.model_provider.input_transformations[i](rlcs_1, position_ids)
            # 3 * [seq_len, k, n_heads, head_dim]
            residual_rlcs_proj = self.model_provider.input_transformations[i](rlcs_0, position_ids, only_affine=True, only_projection=True)
            residual_rlcs_tran = self.model_provider.input_transformations[i](rlcs_1, position_ids, only_affine=True)

            # The model provider obfuscates the random linear combinations (RLCS)
            qkv_rlcs_proj = self.model_provider.obfuscators[i].forward_pass(qkv_rlcs_proj, "qkv")
            qkv_rlcs_tran = self.model_provider.obfuscators[i].forward_pass(qkv_rlcs_tran, "qkv")
            residual_rlcs_proj = self.model_provider.obfuscators[i].forward_pass(residual_rlcs_proj, "attn_out")
            residual_rlcs_tran = self.model_provider.obfuscators[i].forward_pass(residual_rlcs_tran, "attn_out")
            
            # The model provider sends the RLCS to the i-th obfuscator
            self.network_simulator.transfer([qkv_rlcs_proj, qkv_rlcs_tran, residual_rlcs_proj, residual_rlcs_tran], f"forward embedding: M -> O{i}")

            # The i-th obfusctor obfuscate RLCS
            qkv_rlcs_proj = self.obfuscators[i].forward_pass(qkv_rlcs_proj, "qkv")
            qkv_rlcs_tran = self.obfuscators[i].forward_pass(qkv_rlcs_tran, "qkv")
            residual_rlcs_proj = self.obfuscators[i].forward_pass(residual_rlcs_proj, "attn_out")
            residual_rlcs_tran = self.obfuscators[i].forward_pass(residual_rlcs_tran, "attn_out")

            # The i-th obfuscator sends the RLCS to the i-th computation node
            self.network_simulator.transfer([qkv_rlcs_proj, qkv_rlcs_tran, residual_rlcs_proj, residual_rlcs_tran], f"forward embedding: O{i} -> C{i}")

            # The computation node recovers the embeddings
            forward_embedding_share_0 = [reconstruct_random_linear_combination(
                rlc.view(*rlc.shape[:-2], -1), random_transformation[1]).view(*rlc.shape[:-3], 1, GLM6BConfig.n_attention_heads, GLM6BConfig.model_dim // GLM6BConfig.n_attention_heads) for rlc in qkv_rlcs_proj]
            forward_embedding_share_1 = qkv_rlcs_tran
            # (2 * 3(qkv)) * [seq_len, 1, n_heads, head_dim]
            forward_embedding = [a + b for a, b in zip(forward_embedding_share_0, forward_embedding_share_1)]

            residual_embedding_share_0 = reconstruct_random_linear_combination(residual_rlcs_proj, random_transformation[1])
            residual_embedding_share_1 = residual_rlcs_tran
            residual_embedding = residual_embedding_share_0 + residual_embedding_share_1
            # 2 * [seq_len, 1, model_dim]


            # Computation node perform forward computation
            embedding = self.computation_nodes[i].forward_pass(forward_embedding, attention_mask, residual_embedding)

            # When it is the final layer
            if i == len(self.obfuscators) - 1:
                random_transformation = generate_random_transformations(seq_len, self.config.n_random_vectors_final, ensure_sum_one=False)
                random_transformation = random_transformation[0].half().to(self.config.device), random_transformation[1].half().to(self.config.device)
                rlcs = generate_random_linear_combination(embedding, random_transformation[0])
                
                # Computation node sends output RLCS to obfuscator
                self.network_simulator.transfer(rlcs, f"final output embedding: C{i} -> O{i}")
                rlcs = self.obfuscators[i].forward_pass(rlcs, "mlp_output", reverse=True)

                # Obfuscator sends output RLCS to model provider
                self.network_simulator.transfer(rlcs, f"output embedding: O{i} -> M")
                rlcs = self.model_provider.obfuscators[i].forward_pass(rlcs, "mlp_output", reverse=True)
                return rlcs, random_transformation[1]

            else:
                embedding_share_0 = 2 * torch.std(embedding) * (torch.rand_like(embedding) - 0.5)
                embedding_share_1 = embedding - embedding_share_0
                
                random_transformation = generate_random_transformations(seq_len, self.config.n_random_vectors_hidden, ensure_sum_one=False)
                random_transformation = random_transformation[0].half().to(self.config.device), random_transformation[1].half().to(self.config.device)
                rlcs_0 = generate_random_linear_combination(embedding_share_0, random_transformation[0])
                rlcs_1 = embedding_share_1
                
                # Computation node sends output RLCS to obfuscator
                self.network_simulator.transfer([rlcs_0, rlcs_1], f"output embedding: C{i} -> O{i}")
                rlcs_0 = self.obfuscators[i].forward_pass(rlcs_0, "mlp_output", reverse=True)
                rlcs_1 = self.obfuscators[i].forward_pass(rlcs_1, "mlp_output", reverse=True)

                # Obfuscator sends output RLCS to model provider
                self.network_simulator.transfer([rlcs_0, rlcs_1], f"output embedding: O{i} -> M")
                rlcs_0 = self.model_provider.obfuscators[i].forward_pass(rlcs_0, "mlp_output", reverse=True)
                rlcs_1 = self.model_provider.obfuscators[i].forward_pass(rlcs_1, "mlp_output", reverse=True)

    def get_next_token_id(self, rlcs: torch.Tensor, key: torch.Tensor):
        """
        rlcs: [seq_len, n_random_vectors, model_dim]
        """
        random_embeddings = rlcs[-1]
        random_logits = self.glm6b.condgen.lm_head(random_embeddings)[:, :self.glm6b.n_tokens]
        # [n_random_vectors, n_tokens]
        random_logits = random_logits.cpu().numpy()
        key = key.cpu().numpy()
        key_cts = [self.ckks_scheme.enc_vector([k] * self.glm6b.n_tokens) for k in key[-1, 0]]
        random_score_cts = [key_cts[i].mul(random_logits[i]) for i in range(random_logits.shape[0])]
        recovered_score_ct = sum([r for r in random_score_cts[1:]], start=random_score_cts[0])
    
        recovered_score_pts = recovered_score_ct.decrypt()
        token_id = np.argmax(recovered_score_pts)
        return token_id


    def generate(self, query: str, max_len: int=200):
        eos_token_id = self.glm6b.condgen.generation_config.eos_token_id
        input_ids, position_ids, attention_mask = self.glm6b.get_tokenization(query)
        input_len = len(input_ids[0])
        input_ids = input_ids[0].tolist()
        all_ids = input_ids.copy()
        for i in range(max_len - input_len):
            position_ids = position_ids.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)
            e0, e1 = self.retrieve_word_embedding(input_ids)
            prediction_rlcs, prediction_key = self.main_body_inference(e0, e1, position_ids, attention_mask)
            next_token_id = self.get_next_token_id(prediction_rlcs, prediction_key)
            print(next_token_id, self.glm6b.decode(next_token_id))
            all_ids.append(next_token_id.item())
            if next_token_id == eos_token_id:
                break

            current_len = input_len + i + 1
            input_ids = [next_token_id.item()]
            position_ids = generate_position_ids(input_len, current_len)[:, :, current_len - 1:].to(self.config.device)
            attention_mask = generate_attention_mask(input_len, current_len)[:, :, current_len - 1:, :].to(self.config.device)
        return self.glm6b.decode(all_ids)


if __name__ == "__main__":
    network_simulator = NetworkSimulator(0.01, 100 * 1024 * 1024)
    desi_llm = DesiLLM(network_simulator, DesiLLMConfig("cuda", 10, 0.5, 10, 0, 10, 0))
    token_ids, position_ids, attention_mask = desi_llm.glm6b.get_tokenization("Hello, who are you?")
    token_ids = token_ids[0].tolist()
    e0, e1 = desi_llm.retrieve_word_embedding(token_ids)
    result_rlcs, key = desi_llm.main_body_inference(e0, e1, position_ids, attention_mask)
    desi_llm.get_next_token_id(result_rlcs, key)

