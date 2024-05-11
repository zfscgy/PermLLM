def generate_scale_dict(scale_factor: float):
    """
    mask_scale = scale_factor * [maximum scale of the tensor to mask]
    """
    scale_dict = {
        "embedding_retrieval/onehot_matmul/u": 0.01 * scale_factor,
        "embedding_retrieval/onehot_matmul/v": scale_factor,
        "embedding_retrieval/onehot_matmul/w": 0.01 * scale_factor,
        "embedding_retrieval/layernorm_in/x": 0.01 * scale_factor,
        "embedding_retrieval/layernorm_in/z": scale_factor
    }

    for i in range(28):
        prefix = f"transformer_layer_{i}"
        scale_dict.update({
            prefix + "/qkv/u": 0.02 * scale_factor,
            prefix + "/qkv/v": scale_factor,
            prefix + "/qkv/w": 2 * scale_factor,
            
            prefix + "/dot_product/u": 2 * scale_factor,
            prefix + "/dot_product/v": 2 * scale_factor,
            prefix + "/dot_product/w": 5 * scale_factor, 

            prefix + "/softmax/x": 5 * scale_factor,
            prefix + "/softmax/z": scale_factor,

            prefix + "/weighted_sum/u": 2 * scale_factor,
            prefix + "/weighted_sum/v": scale_factor,
            prefix + "/weighted_sum/w": 2 * scale_factor,

            prefix + "/attn_out/u": 0.02 * scale_factor,
            prefix + "/attn_out/v": 2 * scale_factor,
            prefix + "/attn_out/w": 5 * scale_factor,

            prefix + "/layernorm_in/x": 10 * scale_factor,
            prefix + "/layernorm_in/z": scale_factor,

            prefix + "/gelu/x": 1.5 * scale_factor,
            prefix + "/gelu/z": 1.5 * scale_factor,

            prefix + "/layernorm_out/x": 10 * scale_factor,
            prefix + "/layernorm_out/z": scale_factor
        })
    
    scale_dict.update({
        "prediction/final_dense/v": scale_factor,
        "prediction/final_dense/w": 2 * scale_factor,
        "prediction/score_permutation": 2 * scale_factor
    })

    return scale_dict