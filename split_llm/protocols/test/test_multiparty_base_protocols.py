import time

import threading

import torch

from simple_socket.zf_socket import SocketServer
from split_llm.common.communication import Node
from split_llm.common.real_communication import RealCommunication
from split_llm.protocols.base_protocols import SS_Mul__CX_N0_Y_N1
from split_llm.common.utils import test_func


@test_func
def test__SS_Mul__CX_N0_Y_N1():
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
    
    comm0 = RealCommunication(["n0", "n1", "n2"], {"n0": sock0})
    comm1 = RealCommunication(["n0", "n1", "n2"], {"n1": sock1})
    comm2 = RealCommunication(["n0", "n1", "n2"], {"n2": sock2})

    n0 = Node(comm0, "n0")
    n1 = Node(comm1, "n1")
    n2 = Node(comm2, "n2")

    x = torch.tensor([[1, 2]]).float()
    y = torch.tensor([[2], [1]]).float()
    
    protocol_name = "ss_mul__cx_n0_y_n1"

    protocol0 = SS_Mul__CX_N0_Y_N1([1, 2], torch.matmul, protocol_name, n0, Node.from_remote_name("n1"), Node.from_remote_name("n2"), 10)
    protocol1 = SS_Mul__CX_N0_Y_N1([1, 2], torch.matmul, protocol_name, Node.from_remote_name("n0"), n1, Node.from_remote_name("n2"), 10)
    protocol2 = SS_Mul__CX_N0_Y_N1([1, 2], torch.matmul, protocol_name, Node.from_remote_name("n0"), Node.from_remote_name("n1"), n2, 10)

    n0.storage[f"{protocol_name}:x"] = x
    n1.storage[f"{protocol_name}:y"] = y

    prepare_th1 = threading.Thread(target=protocol1.prepare)
    prepare_th2 = threading.Thread(target=protocol2.prepare)
    prepare_th1.start()
    prepare_th2.start()
    protocol0.prepare()
    prepare_th1.join()
    prepare_th2.join()

    offline_th1 = threading.Thread(target=protocol1.offline_execute, args=([2, 1], [1, 1]))
    offline_th2 = threading.Thread(target=protocol2.offline_execute, args=([2, 1], [1, 1]))
    offline_th1.start()
    offline_th2.start()
    protocol0.offline_execute([2, 1], [1, 1])
    offline_th1.join()
    offline_th2.join()


    online_th1 = threading.Thread(target=protocol1.online_execute)
    online_th2 = threading.Thread(target=protocol2.online_execute)
    online_th1.start()
    online_th2.start()
    protocol0.online_execute()
    online_th1.join()
    online_th2.join()


    print("-------------Output Expected/Executed--------------")
    print(x @ y)
    print(n0.storage[f"{protocol_name}:z0"] + n1.storage[f"{protocol_name}:z1"])


if __name__ == "__main__":
    test__SS_Mul__CX_N0_Y_N1()