import copy
import threading
import time

import torch

from simple_socket.zf_socket import SocketServer
from perm_llm.common.communication import Node
from perm_llm.common.real_communication import RealCommunication
from perm_llm.common.utils import test_func
from perm_llm.common.torch_utils import relative_error
from perm_llm.glm6b.wrapped_layer import Attention_GLM_Wrapped, FeedForward_GLM_Wrapped
from perm_llm.glm6b.utils import generate_position_ids
from perm_llm.glm6b.secure_inference import GLMConfig, GLM_TransformerLayerProtocol

import sys
try:
    node_id = int(sys.argv[1])
except:
    node_id = 0

device = "cuda:0"


GLMConfig.model_dim = 768
GLMConfig.head_dim = 64
GLMConfig.n_heads = 12



def test_layer(length: int = 1):
    # Initialize transformer layers

    attn_wrapped = Attention_GLM_Wrapped(GLMConfig.model_dim, GLMConfig.n_heads, 0).to(device)
    attn_wrapped.requires_grad_(False)
    ff_wrapped = FeedForward_GLM_Wrapped(GLMConfig.model_dim, GLMConfig.n_heads, 0).to(device)
    ff_wrapped.requires_grad_(False)
    
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
        node.space.attentions = [attn_wrapped]
    else:
        node.space.attentions = [attn_wrapped]

    if node_id != 2:
        node.space.ffs = [ff_wrapped]


    protocol_name = "transformer_layer"

    if node_id == 0:
        x0 = torch.normal(0, 1, [1, 1, GLMConfig.model_dim]).to(device)
        node.storage[f"{protocol_name}:x0"] = x0
    if node_id == 1:
        node.storage[f"{protocol_name}:x1"] = torch.normal(0, 1, [length, 1, GLMConfig.model_dim]).to(device)

    all_nodes = [Node.from_remote_name("n0"), Node.from_remote_name("n1"), Node.from_remote_name("n2")]
    all_nodes[node_id] = node

    protocol = GLM_TransformerLayerProtocol(*all_nodes, 0, 10, name=protocol_name, device=device)

    print("Start prepare...")

    comm.new_stage("Prepare")

    start_time = time.time()
    protocol.prepare()
    print(f"Prepare stopped in {time.time() - start_time:.3}s.")


    comm.new_stage("offline")

    print("Start offline execute...")
    for i in range(6):
        start_time = time.time()
        protocol.offline_execute(length)
        print(f"Offline execution stopped in {time.time() - start_time:.3}s.")

    # input("Press any key to start online...")
    # TC command can be used here to measure the 

    print("Start online execute (warm up)...")
    comm.new_stage("online")
    start_time = time.time()
    protocol.online_execute()
    print(f"Online execution stopped in {time.time() - start_time:.3}s.")

    print("Start online execute (real test, wait for 3 seconds)...")
    if node_id == 0:
        time.sleep(3)  # This is to wait for the other node to finish the offline phase. (Otherwise the total time could be larger)
    
    # comm.simulate_network(10, 100)  
    # Only uncomment it in cloud server where `tc` command is not permitted.

    for i in range(5):
        comm.new_stage("online")
        start_time = time.time()
        protocol.online_execute()
        print(f"Online execution stopped in {time.time() - start_time:.3}s.")

        print("-------------Output --------------")

        history0 = comm.comm_history['online']
        print(f"Total rounds: {len(history0)}")
        total_bytes = sum([h['size'] for h in history0])
        print(f"Total Mbs: {total_bytes / (1024**2):.2f}Mb")


if __name__ == "__main__":
    test_layer()

"""
To test this file:
First, open 3 terminals and add PYTHONPATH
    (example)
    export PYTHONPATH=/root/autodl-tmp/PermLLM
    export PYTHONPATH=/home/zf/projects/PermLLM


Second, execute them one by one:
    python secure_inference_test__multiparty_layer_processing.py 0
    python secure_inference_test__multiparty_layer_processing.py 1
    python secure_inference_test__multiparty_layer_processing.py 2
"""