import numpy as np
import torch


from split_llm.common.torch_utils import inverse_permutation


def generate_position_ids(input_seq_len: int, current_total_len: int):
    """
    Return: [1, 2, total_len]
    """
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    [x1, x2, x3, x4, x5, x6] -> [-x4, -x5, -x6, x1, x2, x3]
    """
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1) 


def gelu_openai(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


if __name__ == "__main__":
    print(generate_attention_mask(3, 5))
    