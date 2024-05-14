from typing import Any
from dataclasses import dataclass

import time
import pickle
import threading

import torch
import numpy as np
from typing import List, Dict
from simple_socket.zf_socket import SocketServer
from perm_llm.common.utils import test_func
from perm_llm.common.communication import Communication


@dataclass
class WrappedTorchTensor:
    data: np.ndarray
    dtype: torch.dtype

class RealCommunication(Communication):
    def __init__(self, socket_server_map: Dict[str, SocketServer], tensor_device: str="cpu"):
        """
        The socket server is expected to be connected
        """
        self.socket_server_map = socket_server_map

        self.pending_blocking_sends: Dict[str, threading.Thread] = dict()

        self.current_stage = -1
        self.stage_names = []
        self.comm_history: Dict[str, List[dict]] = dict()
        

        self.tensor_device = tensor_device

        # Those two fields are used to add additional time to the send method
        # Using it only for LAN! And watch out for timeout.
        self.latency_ms = None
        self.bandwidth_mbps = None
        self.new_stage("Default")

    def new_stage(self, name: str):
        self.current_stage += 1
        self.stage_names.append(name)
        self.comm_history[name] = []

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

    def simulate_network(self, latency_ms: float, bandwith_mbps: float):
        self.latency_ms = latency_ms
        self.bandwidth_mbps = bandwith_mbps

    def unset_simulation(self):
        self.latency_ms = None
        self.bandwidth_mbps = None

    def send(self, from_role: str, to_role: str, message: Any, header: str):
        wrapped_message = (header, self.wrap_object(message))
        dumped = pickle.dumps(wrapped_message)

        # Simulate the latency and bandwith when in LAN
        if self.latency_ms is not None:
            time.sleep(self.latency_ms / 1000)
        if self.bandwidth_mbps is not None:
            time.sleep(len(dumped) / ((self.bandwidth_mbps * 1024 ** 2) / 8))

        def blocking_send():
            self.socket_server_map[from_role].send_to(to_role, dumped)
            self.comm_history[self.stage_names[-1]].append({
                "from": from_role,
                "to": to_role,
                "time": time.time(),
                "header": header,
                "size": len(dumped)
            })

        if to_role in self.pending_blocking_sends:
            self.pending_blocking_sends[to_role].join()
            del self.pending_blocking_sends[to_role]

        send_th = threading.Thread(target=blocking_send)
        send_th.start()
        self.pending_blocking_sends[to_role] = send_th


    def fetch(self, to_role: str, from_role: str, header: str):
        msg_data = self.socket_server_map[to_role].recv_from(from_role)
        received_header, wrapped_obj = pickle.loads(msg_data)
        if received_header != header:
            raise AssertionError(f"Unexpected header {received_header}, expecting {header}")
        self.comm_history[self.stage_names[-1]].append({
            "from": from_role,
            "to": to_role,
            "time": time.time(),
            "header": header,
            "size": len(msg_data)
        })
        return self.unwrap_object(wrapped_obj)



