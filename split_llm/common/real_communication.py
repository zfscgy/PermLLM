from typing import Any
from dataclasses import dataclass

import time
import pickle
import threading

import torch
import numpy as np
from typing import List, Dict
from simple_socket.zf_socket import SocketServer
from split_llm.common.utils import test_func
from split_llm.common.communication import Communication


@dataclass
class WrappedTorchTensor:
    data: np.ndarray
    dtype: torch.dtype



class RealCommunication(Communication):
    def __init__(self, roles: List[str], socket_server_map: Dict[str, SocketServer], tensor_device: str="cpu"):
        """
        The socket server is expected to be connected
        """
        self.roles = roles
        self.socket_server_map = socket_server_map

        self.current_stage = -1
        self.stage_names = []
        self.comm_history: List[List[dict]] = []

        self.tensor_device = tensor_device

    def wrap_object(self, obj: Any):
        """
        To wrap an object to a serializable format.
        Currently, only copy torch.Tensor to numpy.ndarray
        """
        if isinstance(obj, list) or isinstance(obj, tuple):
            return [self.wrap_object(o) for o in obj]
        elif isinstance(obj, torch.Tensor):
            return WrappedTorchTensor(obj.cpu().numpy(), obj.dtype)
        else:
            return obj

    def unwrap_object(self, obj: Any):
        """
        The reverse process of wrap_object
        """
        if isinstance(obj, list) or isinstance(obj, tuple):
            return [self.unwrap_object(o) for o in obj]
        elif isinstance(obj, WrappedTorchTensor):
            return torch.tensor(obj.data, dtype=obj.dtype, device=self.tensor_device)
        else:
            return obj

    def new_stage(self, name: str):
        self.current_stage += 1
        self.stage_names.append(name)
        self.comm_history.append([])

    def send(self, from_role: str, to_role: str, message: Any, header: str):
        wrapped_message = (header, self.wrap_object(message))
        dumped = pickle.dumps(wrapped_message)
        self.socket_server_map[from_role].send_to(to_role, dumped)

        self.comm_history.append({
            "from": from_role,
            "to": to_role,
            "time": time.time(),
            "header": header,
            "size": len(dumped)
        })

    def fetch(self, to_role: str, from_role: str, header: str):
        received_header, wrapped_obj = pickle.loads(self.socket_server_map[to_role].recv_from(from_role))
        if received_header != header:
            raise AssertionError(f"Unexpected header {received_header}, expecting {header}")
        return self.unwrap_object(wrapped_obj)


if __name__ == "__main__":

    @test_func
    def test_send_receive():
        sock0 = SocketServer("127.0.0.1:4001", {"127.0.0.1:4002": "p1"}, timeout=10)
        sock1 = SocketServer("127.0.0.1:4002", {"127.0.0.1:4001": "p0"}, timeout=10)
        
        time.sleep(1) # Wait the server to start listening

        sock0.connect_all()
        sock1.connect_all()
        
        comm0 = RealCommunication(["p0", "p1"], {"p0": sock0})
        comm1 = RealCommunication(["p0", "p1"], {"p1": sock1})

        send_th = threading.Thread(target=comm0.send, args=("p0", "p1", 1926.0817 * torch.ones(4096), "tensor"))
        send_th.start()
        tensor = comm1.fetch("p1", "p0", "tensor")
        send_th.join()

        print(tensor)
        print(comm0.comm_history)

    test_send_receive()

