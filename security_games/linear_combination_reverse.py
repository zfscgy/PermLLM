import numpy as np
import torch
import tqdm
from desi_llm.common.utils import generate_random_linear_combination
from llm_bases.chatglm6b import ChatGML6B

chatglm6b = ChatGML6B()

def get_word_embedding(word: str):
    token_id = chatglm6b.tokenizer(word)['input_ids'][0]
    embedding = chatglm6b.condgen.transformer.word_embeddings.weight[token_id]
    embedding = embedding.float().cuda()
    return embedding

def test_recover_word():
    embedding0 = get_word_embedding("Eight")
    embeddings = list(map(get_word_embedding, ["One", "Two", "Three", "Five", "Hundred", "Thousand", "Amount", "Math", "Number"]))
    input_embeddings = torch.stack([embedding0])
    input_embeddings += torch.normal(0, 0.008, input_embeddings.shape, device=input_embeddings.device, dtype=input_embeddings.dtype)
    random_vecs, random_coefs = generate_random_linear_combination(input_embeddings, 10)
    # [n, 4096], [n]
    # random_vecs += torch.normal(0, 0.2, random_vecs.shape, device=random_vecs.device, dtype=random_vecs.dtype)
    # print(torch.dist(random_vecs.T @ input_embeddings, random_vecs.T @ torch.stack([embedding0])))
    least_square_errors = []
    for i in tqdm.tqdm(range(chatglm6b.n_tokens)):
        candiate_embedding = chatglm6b.condgen.transformer.word_embeddings.weight.data[i].float().to(random_vecs.device)
        solution = torch.linalg.lstsq(random_vecs.T, candiate_embedding[:, None])
        least_square_errors.append(torch.dist(random_vecs.T @ solution[0], candiate_embedding[:, None]).item())
    # Check the goodness of the solution

    asc_indices = np.argsort(least_square_errors)
    for i in asc_indices[:300]:
        print(f"{least_square_errors[i]:.4f}\t{i}\t{chatglm6b.tokenizer.decode(i)}")


if __name__ == "__main__":
    test_recover_word()