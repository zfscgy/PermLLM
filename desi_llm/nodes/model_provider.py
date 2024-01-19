from ast import main
from typing import List
from IPython import embed

import numpy as np
from sympy import Mod


from simple_pir.pir import PIRServer, PIRClient

from llm_bases.chatglm6b import ChatGML6B
from desi_llm.common.utils import random_vec_with_seed
from desi_llm.glm6b.configs import GLM6BConfig
from desi_llm.glm6b.obfuscated_layer import WrappedGLMBlock, GLMBlockInputTransform, random_orthogonal
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

        self.obfuscation_nodes: List[ObfuscatorNode] = []
        for layer in self.original_layers:
            self.obfuscation_nodes.append(ObfuscatorNode())

        self.shared_rotated_word_embedding: np.ndarray = None
        self.shared_indices: np.ndarray = None
        self.word_embedding_key: np.ndarray = None

        self.pir_server: PIRServer = None


    def generate_obfuscations(self):
        obfuscated_layers = []
        for obfuscator, layer in zip(self.obfuscation_nodes, self.original_layers):
            obfuscated_layers.append(obfuscator.obfuscate(layer))
        return obfuscated_layers

    def generate_shared_embedding(self) -> np.ndarray:
        """
        return tuple: 
            * random_share_obfuscator   for obfuscator: the share of rotated word embeddings 
            * (lwe_mat, hint)    for user: the lwe matrix and the hint for PIR

        """
        self.word_embedding_key = random_orthogonal(GLM6BConfig.model_dim)
        # Extract embedding to numpy
        embedding = self.full_model.condgen.transformer.word_embeddings.weight[:ChatGML6B.max_token_id].numpy().astype(np.float32)  # [vocab_size, model_dim]
        
        # Rotate the embedding
        rotated_embedding = embedding @ self.word_embedding_key.T

        # Random permutation/Additive sharing
        permutation = np.random.permutation(ChatGML6B.max_token_id)
        random_share_server = np.zeros_like(embedding)
        for i in range(ChatGML6B.max_token_id):
            random_share_server[i] = random_vec_with_seed(permutation[i], GLM6BConfig.model_dim, [-1, 1])
        random_share_obfuscator = rotated_embedding - random_share_server
        permuted_share_obfuscator = torch.zeros_like(random_share_obfuscator)
        permuted_share_obfuscator[permutation] = random_share_obfuscator
        self.shared_rotated_word_embedding = random_share_server
        self.permutation = permutation
        return permuted_share_obfuscator

def setup_pir_server(self):
    # Build PIR Server
    pir_db1 = self.permutation // 1000
    pir_db2 = self.permutation % 1000
    self.pir_server = PIRServer(pir_db1.tolist() + pir_db2.tolist())
    return self.pir_server.lwe_mat, self.pir_server.setup()


if __name__ == "__main__":
    def test_generate_shared_embedding():
        model_provider = ModelProvider(ChatGML6B())
        random_share_1 = model_provider.generate_shared_embedding()
        token_id = 130004
        embedding1 = model_provider.shared_rotated_word_embedding[token_id] + random_share_1[model_provider.permutation[token_id]]
        embedding1 = embedding1 @ model_provider.word_embedding_key
        embedding0 = model_provider.full_model.condgen.transformer.word_embeddings.weight[token_id]
        print(embedding1)
        print(embedding0)


    test_generate_shared_embedding()
