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
from split_llm.common.utils import test_func
from split_llm.common.real_communication import RealCommunication


@test_func
def receive():
    sock1 = SocketServer("127.0.0.1:4002", {"127.0.0.1:4001": "p0"}, timeout=10)
    time.sleep(10) # Wait the server to start listening
    sock1.connect_all()
    comm1 = RealCommunication(["p0", "p1"], {"p1": sock1})
    tensor = comm1.fetch("p1", "p0", "tensor")
    print(tensor)
    print(comm1.socket_server_map["p1"].traffic_counter_from)


if __name__ == "__main__":
    receive()