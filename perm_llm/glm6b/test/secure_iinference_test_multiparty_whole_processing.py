import copy
import threading
import time

import torch
import os
from perm_llm.glm6b.wrapped_layer import Attention_GLM_Wrapped, copy_attention, FeedForward_GLM_Wrapped, copy_feedforward
from perm_llm.glm6b.utils import generate_position_ids

from typing import List

from llm_bases.chatglm6b import ChatGML6B
glm = ChatGML6B()

from llm_bases.chatglm6b_official.modeling_chatglm import GLMBlock

from simple_socket.zf_socket import SocketServer
from perm_llm.common.communication import Node
from perm_llm.common.real_communication import RealCommunication
from perm_llm.common.utils import test_func
from perm_llm.common.torch_utils import relative_error
from perm_llm.glm6b.wrapped_layer import Attention_GLM_Wrapped, FeedForward_GLM_Wrapped
from perm_llm.glm6b.utils import generate_position_ids
from perm_llm.glm6b.secure_inference import GLM_Protocol
from perm_llm.glm6b.secure_inference_utils import generate_scale_dict

import sys
try:
    node_id = int(sys.argv[1])
except:
    node_id = 0

def test_whole(length: int = 1):
    device = ["cuda:0", "cuda:0", "cuda:0"][node_id]

    # Initialize transformer layers
    raw_glm_layers: List[GLMBlock] = glm.condgen.transformer.layers
    attentions: List[Attention_GLM_Wrapped] = []
    ffs: List[FeedForward_GLM_Wrapped] = []
    for i in range(28):
        transformer_layer = raw_glm_layers[i].float()
        
        # The private attention layer
        attn_wrapped = Attention_GLM_Wrapped(4096, 32, i)
        copy_attention(transformer_layer, attn_wrapped)
        attn_wrapped.requires_grad_(False)
        if node_id != 0:
            attn_wrapped.qkv_weight = attn_wrapped.qkv_bias = None
            attn_wrapped.attn_out_weight = attn_wrapped.attn_out_bias = None
        attentions.append(attn_wrapped.to(device))

        ff_wrapped = FeedForward_GLM_Wrapped(4096, 32, i)
        if node_id != 0:
            ff_wrapped.mlp_dense_in = None
            ff_wrapped.mlp_dense_out = None
        if i == 27:
            copy_feedforward(transformer_layer, None, ff_wrapped)
            ff_wrapped.layernorm_out = glm.condgen.transformer.final_layernorm.float()
        else:
            copy_feedforward(transformer_layer, raw_glm_layers[i + 1].float(), ff_wrapped)
        ff_wrapped.requires_grad_(False)
        ffs.append(ff_wrapped.to(device))
    
    input_layernorm = raw_glm_layers[0].input_layernorm.float().to(device)

    if node_id == 0:
        word_embedding = glm.condgen.transformer.word_embeddings.weight.float().to(device)

    print(f"Current node: {node_id}")

    # Set up communication
    address_dict = {
        "127.0.0.1:6000": "n0",
        "127.0.0.1:6001": "n1",
        "127.0.0.1:6002": "n2"
    }

    listen_address = f"127.0.0.1:600{node_id}"
    sock = SocketServer(listen_address, address_dict)
    
    time.sleep(10) # Wait the server to start listening
    sock.connect_all()
    print("Socket connected")

    comm = RealCommunication({f"n{node_id}": sock}, tensor_device=device)
    node = Node(comm, f"n{node_id}")

    # Delete sensitive weights
    

    if node_id != 0:
        attn_wrapped.qkv_weight = None
        attn_wrapped.qkv_bias = None
        attn_wrapped.attn_out_weight = None
        attn_wrapped.attn_out_bias = None
    else:
        node.space.word_embedding = word_embedding
    
    node.space.attentions = [attn_wrapped]

    if node_id != 2:
        node.space.ffs = [ff_wrapped]
        node.space.input_layernorm = input_layernorm


    all_nodes = [Node.from_remote_name("n0"), Node.from_remote_name("n1"), Node.from_remote_name("n2")]
    all_nodes[node_id] = node
    protocol = GLM_Protocol(*all_nodes, generate_scale_dict(100), device=device)
    
    input("Model load successfully, press any key to start prepare...")

    comm.new_stage("Prepare")

    start_time = time.time()
    protocol.prepare()
    print(f"Prepare stopped in {time.time() - start_time:.3}s.")


    comm.new_stage("offline")

    print("Start offline execute...")
    for length in [5] + [1] * 20:
        start_time = time.time()
        protocol.offline_execute(length)
        print(f"Offline execution stopped in {time.time() - start_time:.3}s.")

    # input("Press any key to start online...")
    # TC command can be used here to measure the 

    
    print("Start online execute (real test, wait for 3 seconds)...")
    if node_id == 1:
        query = "Tell me about Trump"  # After tokenization, the length shall be 6
        input_ids, _, _ = glm.get_tokenization(query)
        input_ids = input_ids[0]
        input_selector = torch.zeros(len(input_ids), glm.n_tokens)
        for i in range(len(input_ids)):
            input_selector[i, input_ids[i]] = 1

    for i in range(21):

        if node_id == 1:
            print("Predicted token: ", glm.decode(node.storage[f"{protocol.name}:z"]))
            if i == 0:
                input_tensor = input_selector[:-1]
            node.storage[f"{protocol.name}:x"] = input_tensor
        comm.new_stage("online")
        start_time = time.time()
        protocol.online_execute()
        print(f"Online execution stopped in {time.time() - start_time:.3}s.")

        print("-------------Output --------------")

        history0 = comm.comm_history['online']
        print(f"Total rounds: {len(history0)}")
        total_bytes = sum([h['size'] for h in history0])
        print(f"Total Mbs: {total_bytes / (1024**2):.2f}Mb")

        if node_id == 1:
            print("Predicted token: ", glm.decode(node.storage[f"{protocol.name}:z"]))
            if i == 0:
                input_tensor = input_selector[:-1]
            else:
                input_tensor = torch.zeros([1, glm.n_tokens])
                input_tensor[0, node.storage[f"{protocol.name}:z"]] = 1



if __name__ == "__main__":
    test_whole()