import copy
import threading
import time

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)-8s %(message)s")

import torch

from simple_socket.zf_socket import SocketServer
from perm_llm.common.communication import Node
from perm_llm.common.real_communication import RealCommunication
from perm_llm.common.utils import test_func
from perm_llm.common.torch_utils import relative_error
from perm_llm.glm6b.wrapped_layer import Attention_GLM_Wrapped, FeedForward_GLM_Wrapped
from perm_llm.glm6b.utils import generate_position_ids
from perm_llm.glm6b.secure_inference import GLM_TransformerLayerProtocol


def test_layer():
    device = "cuda"

    # Initialize transformer layers

    attn_wrapped = Attention_GLM_Wrapped(4096, 32, 0).to(device)
    attn_wrapped.requires_grad_(False)
    ff_wrapped = FeedForward_GLM_Wrapped(4096, 32, 0).to(device)
    ff_wrapped.requires_grad_(False)

    print("Start perform local inference")
    xs = torch.normal(0, 1, [1, 1, 4096]).to(device)
    position_ids = generate_position_ids(2, 2)[:, :, :1].to(device)
    

    expected_h = attn_wrapped(xs, position_ids)
    expected_out = ff_wrapped(expected_h + (2 * 28) ** 0.5 * xs)

    print("Computed expected output.\nStart MPC...")


    # Set up communication
    address_dict = {
        "127.0.0.1:4000": "n0",
        "127.0.0.1:4001": "n1",
        "127.0.0.1:4002": "n2"
    }
    sock0 = SocketServer("127.0.0.1:4000", address_dict)
    sock1 = SocketServer("127.0.0.1:4001", address_dict)
    sock2 = SocketServer("127.0.0.1:4002", address_dict)
    
    time.sleep(1) # Wait the server to start listening

    connect1 = threading.Thread(target=sock1.connect_all)
    connect2 = threading.Thread(target=sock2.connect_all)
    connect1.start()
    connect2.start()
    sock0.connect_all()
    connect1.join()
    connect2.join()
    print("All sockets connected")
     
    comm0 = RealCommunication({"n0": sock0}, tensor_device=device)
    comm1 = RealCommunication({"n1": sock1}, tensor_device=device)
    comm2 = RealCommunication({"n2": sock2}, tensor_device=device)

    n0 = Node(comm0, "n0")
    n1 = Node(comm1, "n1")
    n2 = Node(comm2, "n2")

    # Delete sensitive weights
    attn_wrapped_public = Attention_GLM_Wrapped(4096, 32, 0).to(device)
    attn_wrapped_public.qkv_weight = None
    attn_wrapped_public.qkv_bias = None
    attn_wrapped_public.attn_out_weight = None
    attn_wrapped_public.attn_out_bias = None
    attn_wrapped_public.positional_embedding = attn_wrapped.positional_embedding
    attn_wrapped.requires_grad_(False)
    
    
    n0.space.attentions = [attn_wrapped]
    n1.space.attentions = n2.space.attentions = [attn_wrapped_public]
    n0.space.ffs = n1.space.ffs = [ff_wrapped]


    protocol_name = "transformer_layer"
    
    x0 = torch.normal(0, 1, [1, 1, 4096]).to(device)
    x1 = xs - x0
    n0.storage[f"{protocol_name}:x0"] = x0
    n1.storage[f"{protocol_name}:x1"] = x1

    protocol0 = GLM_TransformerLayerProtocol(n0, Node.from_remote_name("n1"), Node.from_remote_name("n2"), 0, 10, private_mlp=False, name=protocol_name, device=device)
    protocol1 = GLM_TransformerLayerProtocol(Node.from_remote_name("n0"), n1, Node.from_remote_name("n2"), 0, 10, private_mlp=False, name=protocol_name, device=device)
    protocol2 = GLM_TransformerLayerProtocol(Node.from_remote_name("n0"), Node.from_remote_name("n1"), n2, 0, 10, private_mlp=False, name=protocol_name, device=device)

    print("Start prepare...")

    comm0.new_stage("Prepare")
    comm1.new_stage("Prepare")
    comm2.new_stage("Prepare")

    start_time = time.time()
    prepare_th1 = threading.Thread(target=protocol1.prepare)
    prepare_th2 = threading.Thread(target=protocol2.prepare)
    prepare_th1.start()
    prepare_th2.start()
    protocol0.prepare()
    prepare_th1.join()
    prepare_th2.join()
    print(f"Prepare stopped in {time.time() - start_time:.3}s.")


    comm0.new_stage("offline")
    comm1.new_stage("offline")
    comm2.new_stage("offline")

    print("Start offline execute...")
    start_time = time.time()
    offline_th1 = threading.Thread(target=protocol1.offline_execute, args=(1,))
    offline_th2 = threading.Thread(target=protocol2.offline_execute, args=(1,))
    offline_th1.start()
    offline_th2.start()
    protocol0.offline_execute(1)
    offline_th1.join()
    offline_th2.join()
    print(f"Offline execution stopped in {time.time() - start_time:.3}s.")

    # input("Press any key to start online...")
    # TC command can be used here to measure the 

    comm0.new_stage("online")
    comm1.new_stage("online")
    comm2.new_stage("online")

    print("Start online execute...")
    start_time = time.time()
    online_th1 = threading.Thread(target=protocol1.online_execute)
    online_th2 = threading.Thread(target=protocol2.online_execute)
    online_th1.start()
    online_th2.start()
    protocol0.online_execute()
    online_th1.join()
    online_th2.join()
    print(f"Online execution stopped in {time.time() - start_time:.3}s.")

    print("-------------Output --------------")
    mpc_out = n0.storage[f"{protocol_name}:z0"] + n1.storage[f"{protocol_name}:z1"]
    print(f"Error rate of mpc: {relative_error(mpc_out, expected_out):.5f}")

    history0 = comm0.comm_history['online']
    print(f"Total rounds: {len(history0)}")
    total_bytes = sum([h['size'] for h in history0])
    print(f"Total Mbs: {total_bytes / (1024**2):.2f}Mb")


if __name__ == "__main__":
    test_layer()