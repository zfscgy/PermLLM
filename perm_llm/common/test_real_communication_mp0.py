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
def send():
    sock0 = SocketServer("127.0.0.1:4001", {"127.0.0.1:4002": "p1"}, timeout=10)
    time.sleep(10) # Wait the server to start listening
    sock0.connect_all()
    comm0 = RealCommunication(["p0", "p1"], {"p0": sock0})
    comm0.send("p0", "p1", 1926.0817 * np.ones((4096, 100000), dtype=np.float32), "tensor")
    print(comm0.comm_history)


if __name__ == "__main__":
    send()