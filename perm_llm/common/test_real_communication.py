from typing import Any
from dataclasses import dataclass

import time
import pickle
import threading
import multiprocessing

from functools import partial

import torch
import numpy as np
from typing import List, Dict
from simple_socket.zf_socket import SocketServer
from perm_llm.common.utils import test_func
from perm_llm.common.real_communication import RealCommunication

@test_func
def test_send_receive():
    sock0 = SocketServer("127.0.0.1:4001", {"127.0.0.1:4002": "p1"}, timeout=15)
    sock1 = SocketServer("127.0.0.1:4002", {"127.0.0.1:4001": "p0"}, timeout=15)
    
    time.sleep(1) # Wait the server to start listening

    sock0.connect_all()
    sock1.connect_all()
    
    comm0 = RealCommunication({"p0": sock0})
    comm1 = RealCommunication({"p1": sock1})

    send_th = threading.Thread(target=comm0.send, args=("p0", "p1", 1926.0817 * torch.ones(4096, 100000), "tensor"))
    send_th.start()
    tensor = comm1.fetch("p1", "p0", "tensor")
    send_th.join()

    print(tensor)
    print(comm0.comm_history)




if __name__ == "__main__":

    test_send_receive()