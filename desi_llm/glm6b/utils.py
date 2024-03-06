import numpy as np
import torch


def generate_position_ids(input_seq_len: int, current_total_len: int):
    p0 = list(range(input_seq_len - 1)) + [input_seq_len - 2] * (current_total_len - input_seq_len + 1)
    p1 = [0] * (input_seq_len - 1) + list(range(1, current_total_len - input_seq_len + 2))
    return torch.tensor(np.array([[p0, p1]]))


def generate_attention_mask(input_seq_len: int, current_total_len: int):
    """
    attention_mask[:, :, pos, :] to get the attention mask on the position
    """
    attention_mask = np.tril(np.ones([1, 1, current_total_len, current_total_len]))
    attention_mask[:, :, :, :input_seq_len] = 1
    return torch.tensor(attention_mask) < 0.5