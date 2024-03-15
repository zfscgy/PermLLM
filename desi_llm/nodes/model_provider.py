from typing import Tuple, List
from IPython import embed

import numpy as np
from sympy import Mod

import torch
import torch.nn.functional as F

from simple_pir.pir import PIRServer, PIRClient

from llm_bases.chatglm6b import ChatGML6B
from desi_llm.common.utils import random_vec_with_seed
from desi_llm.glm6b.configs import GLM6BConfig
from desi_llm.glm6b.obfuscated_layer import WrappedGLMBlock, GLMBlockInputTransform, random_orthonormal
from desi_llm.nodes.obfuscator import ObfuscatorNode



class ModelProvider:
    def __init__(self, full_model: ChatGML6B) -> None:
        self.full_model = full_model
        self.original_layers: List[WrappedGLMBlock] = []
        self.input_transformations: List[GLMBlockInputTransform] = []
        for layer in self.full_model.condgen.transformer.layers:
            self.original_layers.append(WrappedGLMBlock(layer.layer_id))
            self.original_layers[-1].wrap(layer)
            self.input_transformations.append(GLMBlockInputTransform(layer))

        self.obfuscators: List[ObfuscatorNode] = []
        for layer in self.original_layers:
            self.obfuscators.append(ObfuscatorNode())

        self.embedding_share: np.ndarray = None
        self.shared_indices: np.ndarray = None
        self.word_embedding_key: np.ndarray = None

        self.pir_server: PIRServer = None


    def generate_obfuscations(self, device: str = "cuda"):
        obfuscated_layers = []
        for obfuscator, layer in zip(self.obfuscators, self.original_layers):
            obfuscated_layers.append(obfuscator.obfuscate(layer, device))
        return obfuscated_layers

    def generate_shared_embeddings(self, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return tuple: 
            * random_share_obfuscator   for obfuscator: the share of rotated word embeddings 
        """
        embedding = self.full_model.condgen.transformer.word_embeddings.weight[:ChatGML6B.n_tokens].to(device)  # [vocab_size, model_dim]
        embedding = F.layer_norm(embedding.float(), [embedding.shape[1]])  # Layernorm in advance

        # Random permutation/Additive sharing
        self.embedding_perm = torch.randperm(ChatGML6B.n_tokens)
        random_share_0 = torch.zeros_like(embedding)
        
        for i in range(ChatGML6B.n_tokens):
            random_share_0[i] = torch.tensor(random_vec_with_seed(self.embedding_perm[i].item(), GLM6BConfig.model_dim, [-1, 1]), dtype=embedding.dtype, device=embedding.device)

        random_share_1 = torch.rand_like(embedding[self.embedding_perm]) - 2
        random_share_2 = embedding[self.embedding_perm] - random_share_0 - random_share_1

        self.embedding_share = random_share_0
        return random_share_1, random_share_2

    def setup_pir_server(self):
        # Build PIR Server
        pir_db1 = self.embedding_perm // 1000
        pir_db2 = self.embedding_perm % 1000
        pir_db = torch.stack([pir_db1, pir_db2], dim=1).view(-1).tolist()
        self.pir_server = PIRServer(pir_db)
        return self.pir_server.lwe_mat, self.pir_server.generate_hint()


if __name__ == "__main__":
    def test_generate_shared_embedding():
        model_provider = ModelProvider(ChatGML6B())
        random_share_1 = model_provider.generate_shared_embeddings()
        token_id = 130004
        embedding1 = model_provider.rotated_word_embedding_shared[token_id] + random_share_1[model_provider.word_embedding_permutation[token_id]]
        embedding1 = embedding1 @ model_provider.word_embedding_key
        embedding0 = model_provider.full_model.condgen.transformer.word_embeddings.weight[token_id]
        print(embedding1)
        print(embedding0)


    test_generate_shared_embedding()
