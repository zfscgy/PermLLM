import time

from functools import partial
import numpy as np
import torch

from simple_socket.zf_socket import SocketServer
from perm_llm.common.communication import Node
from perm_llm.common.real_communication import RealCommunication

from perm_llm.protocols.element_wise import SS_ElementWise__RandPerm
from perm_llm.common.torch_utils import permute_2d_with_seed

import sys
try:
    node_id = int(sys.argv[1])
except:
    node_id = 0

device = "cpu"



def test_protocol(input_size):
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
    protocol_name = "operation_protocol"


    all_nodes = [Node.from_remote_name("n0"), Node.from_remote_name("n1"), Node.from_remote_name("n2")]
    all_nodes[node_id] = node

    protocol = SS_ElementWise__RandPerm(
        permute_2d_with_seed, partial(permute_2d_with_seed, reverse=True),
        partial(torch.layer_norm, normalized_shape=[input_size[-1]]), protocol_name,
        *all_nodes, 10, device=device)

    print("Start prepare...")

    comm.new_stage("Prepare")

    start_time = time.time()
    protocol.prepare()
    print(f"Prepare stopped in {time.time() - start_time:.3}s.")


    comm.new_stage("offline")

    print("Start offline execute...")
    for i in range(6):
        if node_id == 0:
            node.storage[f"{protocol_name}:new_perm"] = node.storage[f"{protocol_name}:new_invperm"] = i
        start_time = time.time()
        protocol.offline_execute(input_size)
        print(f"Offline execution stopped in {time.time() - start_time:.3}s.")

    # input("Press any key to start online...")
    # TC command can be used here to measure the 

    print("Start online execute (warm up)...")
    comm.new_stage("online")
    start_time = time.time()
    
    if node_id != 2:
        x_share = torch.normal(0, 1, input_size).to(device)
        node.storage[f"{protocol_name}:x{node_id}"] = x_share

    protocol.online_execute()
    print(f"Online execution stopped in {time.time() - start_time:.3}s.")

    print("Start online execute (real test, wait for 3 seconds)...")
    if node_id == 0:
        time.sleep(3)  # This is to wait for the other node to finish the offline phase. (Otherwise the total time could be larger)
    
    # comm.simulate_network(10, 100)  
    # Only uncomment it in cloud server where `tc` command is not permitted.
    times = []

    for i in range(5):
        comm.new_stage("online")
        start_time = time.time()
        protocol.online_execute()
        total_time = time.time() - start_time
        print(f"Online execution stopped in {total_time}s.")
        times.append(total_time)

        print("-------------Output --------------")

        history0 = comm.comm_history['online']
        print(f"Total rounds: {len(history0)}")
        total_bytes = sum([h['size'] for h in history0])
        print(f"Total Mbs: {total_bytes / (1024**2):.2f}Mb")

    print("=====================================")
    print(f"{np.mean(times)} ± {np.std(times)}")


if __name__ == "__main__":
    test_protocol([1, 1000])


"""
To test this file:
First, open 3 terminals and add PYTHONPATH
    (example)
    export PYTHONPATH=/root/autodl-tmp/PermLLM
    export PYTHONPATH=/home/zf/projects/PermLLM


Second, execute them one by one:
    python test_multiparty_base_protocols_processing.py 0
    python test_multiparty_base_protocols_processing.py 1
    python test_multiparty_base_protocols_processing.py 2

"""