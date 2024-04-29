import copy
import threading
import time

import torch

from simple_socket.zf_socket import SocketServer
from split_llm.common.communication import Node
from split_llm.common.real_communication import RealCommunication
from split_llm.common.torch_utils import relative_error
from split_llm.glm6b.wrapped_layer import Attention_GLM_Wrapped, FeedForward_GLM_Wrapped
from split_llm.glm6b.utils import generate_position_ids
from split_llm.glm6b.secure_inference import GLM_TransformerLayerProtocol


def test_layer():
    device = "cuda"

    # Initialize transformer layers

    attn_wrapped = Attention_GLM_Wrapped(4096, 32, 0).to(device)
    attn_wrapped.requires_grad_(False)
    ff_wrapped = FeedForward_GLM_Wrapped(4096, 32, 0).to(device)
    ff_wrapped.requires_grad_(False)

    print("Start perform local inference")
    xs = torch.normal(0, 1, [10, 1, 4096]).to(device)
    position_ids = generate_position_ids(10, 10).to(device)
    

    expected_h = attn_wrapped(xs, position_ids)
    expected_out = ff_wrapped(expected_h + (2 * 28) ** 0.5 * xs)

    print("Computed expected output.\nStart MPC...")


    # Set up communication
    address_dict = {
        "127.0.0.1:9000": "n0",
        "127.0.0.1:9001": "n1",
        "127.0.0.1:9002": "n2"
    }
    sock0 = SocketServer("127.0.0.1:9000", address_dict)
    sock1 = SocketServer("127.0.0.1:9001", address_dict)
    sock2 = SocketServer("127.0.0.1:9002", address_dict)
    
    time.sleep(1) # Wait the server to start listening

    sock0.connect_all()
    sock1.connect_all()
    sock2.connect_all()
    
    comm0 = RealCommunication(["n0", "n1", "n2"], {"n0": sock0}, tensor_device=device)
    comm1 = RealCommunication(["n0", "n1", "n2"], {"n1": sock1}, tensor_device=device)
    comm2 = RealCommunication(["n0", "n1", "n2"], {"n2": sock2}, tensor_device=device)

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
    
    x0 = torch.normal(0, 1, [10, 1, 4096]).to(device)
    x1 = xs - x0
    n0.storage[f"{protocol_name}:x0"] = x0
    n1.storage[f"{protocol_name}:x1"] = x1

    protocol0 = GLM_TransformerLayerProtocol(n0, Node.from_remote_name("n1"), Node.from_remote_name("n2"), 0, 10, 10, name=protocol_name, device=device)
    protocol1 = GLM_TransformerLayerProtocol(Node.from_remote_name("n0"), n1, Node.from_remote_name("n2"), 0, 10, 10, name=protocol_name, device=device)
    protocol2 = GLM_TransformerLayerProtocol(Node.from_remote_name("n0"), Node.from_remote_name("n1"), n2, 0, 10, 10, name=protocol_name, device=device)

    print("Start prepare...")
    start_time = time.time()
    prepare_th1 = threading.Thread(target=protocol1.prepare)
    prepare_th2 = threading.Thread(target=protocol2.prepare)
    prepare_th1.start()
    prepare_th2.start()
    protocol0.prepare()
    prepare_th1.join()
    prepare_th2.join()
    print(f"Prepare stopped in {time.time() - start_time:.3}s.")

    print("Start offline execute...")
    start_time = time.time()
    offline_th1 = threading.Thread(target=protocol1.offline_execute, args=(10,))
    offline_th2 = threading.Thread(target=protocol2.offline_execute, args=(10,))
    offline_th1.start()
    offline_th2.start()
    protocol0.offline_execute(10)
    offline_th1.join()
    offline_th2.join()
    print(f"Offline execution stopped in {time.time() - start_time:.3}s.")
    input("Press any key to start online...")
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
    print(f"Error rate of mpc: {relative_error(mpc_out, expected_out):.4f}")


if __name__ == "__main__":
    test_layer()