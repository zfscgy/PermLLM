from typing import Any, List, Callable
from functools import partial

import numpy as np
import torch
from torch import optim
from llm_bases.chatglm6b_official.modeling_chatglm import ChatGLMForConditionalGeneration
from llm_bases.chatglm6b_official.tokenization_chatglm import ChatGLMTokenizer

from transformers import AutoTokenizer, AutoModel




class ChatGML6B:
    pretrained_model_path: str = "/root/autodl-tmp/chatglm-6b"


    n_tokens: int = 130528  # 130005 is the eos token

    def __init__(self, device: str = "cpu"):
        self.device: str = device
        self.dtype: torch.dtype = torch.half
        self.tokenizer: ChatGLMTokenizer = ChatGLMTokenizer.from_pretrained(self.pretrained_model_path)
        self.condgen: ChatGLMForConditionalGeneration = \
            ChatGLMForConditionalGeneration.from_pretrained(self.pretrained_model_path).float().to(self.device)
        self.condgen.requires_grad_(False)

    def chat(self, query: str):
        return self.condgen.chat(self.tokenizer, query)


    """
    Some customized functions for ChatGLM6B
    1.  Adding noise in the hidden states
    2.  Reconstruct the input given the hidden states
    """
    def get_tokenization(self, query: str):
        """
        Return: input_ids, position_ids, attention_masks
        """
        tokenization = self.tokenizer(query, return_tensors="pt", padding=True).to(self.device)
        return tokenization.input_ids, tokenization.position_ids, tokenization.attention_mask

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def get_initial_state(self, input_ids: torch.Tensor):
        word_embs = self.condgen.transformer.word_embeddings(input_ids)
        return torch.transpose(word_embs, 0, 1)  # Notice the first dimension is the sequence position!


    def forward_layers(self, hidden_state: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor,
                       start_layer: int = 0, end_layer: int = None):
        """
        :param hidden_state:
        :param position_ids:
        :param attention_mask:
        :param start_layer: included
        :param end_layer: exclueded
        :return:
        """
        end_layer = end_layer or len(self.condgen.transformer.layers)
        for i, layer in enumerate(self.condgen.transformer.layers[start_layer:end_layer]):
            hidden_state = layer(
                hidden_state,
                position_ids=position_ids,
                attention_mask=attention_mask,
                layer_id=torch.tensor(i + start_layer)
            )[0]
        return hidden_state

    def generate_next_token(self, input_ids, position_ids, attention_masks):
        """
        Adding noise in one intermediate layer. By default, no noise is added.
        :param query:
        :param split_layer:
        :param noise_std:
        :return:
        """
        hidden_state = self.get_initial_state(input_ids)
        final_emb = self.forward_layers(hidden_state, position_ids, attention_masks)

        logits = self.condgen.lm_head(self.condgen.transformer.final_layernorm(final_emb)).permute(1, 0, 2).contiguous()
        # Get the logits on the next position

        probs = torch.softmax(logits, dim=-1)  # [batch, length, n_tokens]
        return probs[0, -1]

    def greedy_generate(self, query: str, prob_generator: Callable, max_gen_length: int=300):
        input_ids, position_ids, attention_masks = self.get_tokenization(query)
        context_size = len(input_ids[0])
        for i in range(max_gen_length):
            probs = prob_generator(input_ids, position_ids, attention_masks)
            next_id = torch.argmax(probs)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]]).to(self.device)], dim=-1)  # Append the last id
            new_seq_len = len(input_ids[0])
            attention_masks = np.tril(np.ones([1, 1, new_seq_len, new_seq_len]))
            attention_masks[:, :, :context_size - 1, :context_size - 1] = 1
            attention_masks = (torch.tensor(attention_masks) < 0.5).to(self.device)
            position_ids = torch.cat([
                position_ids,
                torch.tensor([[[context_size - 2], [new_seq_len - context_size + 1]]]).to(self.device)
            ], dim=-1)
            if next_id == self.condgen.generation_config.eos_token_id:
                break

        resp = self.tokenizer.decode(input_ids[0, context_size:])
        return resp


if __name__ == '__main__':
    model = ChatGML6B("cuda")

    query = "Tell me about Trump"
    print("Expected:")
    print(model.chat(query)[0])

    print("Implemented:")
    print(model.greedy_generate(query, model.generate_next_token))
    # print(model.chat("你是谁？")[0])
    # print(model.chat("Hello")[0])
    # print(model.chat("Tell me about Trump")[0])



    # test_greedy_generate()
    # test_greedy_generate_with_noise()