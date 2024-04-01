from typing import Any, Tuple, List
import types

import torch
import numpy as np


def estimate_size(m):
    if isinstance(m, np.ndarray):
        if m.dtype == np.float32:
            return m.size * 4
        elif m.dtype in [np.int32, np.int64, np.uint32, np.uint64]:
            bit_length = np.ceil(np.log2(np.max(m) - np.min(m)))
            return m.size * bit_length / 4
        else:
            raise ValueError("Unsupported NumPy type for size estimation")
    if isinstance(m, torch.Tensor):
        if m.dtype == torch.float:
            return np.prod(m.shape) * 4
        elif m.dtype == torch.half:
            return np.prod(m.shape) * 2
        elif m.dtype in [torch.int, torch.long]:
            bit_length = torch.ceil(torch.log2(np.max(m) - torch.min(m))).item()
            return np.prod(m.shape) * bit_length / 4
        else:
            raise ValueError("Unsupported Torch type for size estimation")

    elif isinstance(m, list) or isinstance(m, tuple):
        return sum([estimate_size(mm) for mm in m])
    else:
        raise ValueError("Unsupported type for size estimation")
    


class Communication:
    def __init__(self, roles: List[str]):
        raise NotImplementedError()

    def send(self, from_role: str, to_role: str, message: Any, header: str):
        raise NotImplementedError()


    def fetch(self, to_role: str, from_role: str, header: str):
        raise NotImplementedError()

    def generate_report(self, latency: float, bandwidth: float):
        raise NotImplementedError()



class SimulatedCommunication(Communication):
    def __init__(self, roles: List[str]):
        self.comm_history: List[List[dict]] = []
        self.roles = roles
        self.communication_buffer = dict()
        self.current_stage = -1
        self.stage_names = []

    def new_stage(self, name: str):
        self.current_stage += 1
        self.stage_names.append(name)
        self.comm_history.append([])

    def send(self, from_role: str, to_role: str, message: Any, header: str):
        msg_size = estimate_size(message)
        self.communication_buffer[f"{from_role}-{to_role}-{header}"] = message
        self.comm_history[self.current_stage].append({
            "from": from_role,
            "to": to_role,
            "header": header,
            "size": msg_size
        })
    
    def fetch(self, to_role: str, from_role: str, header: str):
        return self.communication_buffer.pop(f"{from_role}-{to_role}-{header}")

    def generate_report(self, latency: float, bandwidth: float):
        """
        The bandwidth's unit is bytes.
        """
        comms = [0 for _ in range(self.current_stage)]
        times = [0 for _ in range(self.current_stage)]
        for s in range(self.current_stage):
            for history in self.comm_history[s]:
                comms[s] += history['size']
                times[s] += latency + history['size'] / bandwidth
        
        return comms, times


class Node:
    def __init__(self, communication: Communication, name: str) -> None:
        self.comm = communication
        self.name = name
        self.space = types.SimpleNamespace()
        self.storage = dict()

    def send(self, to: str, header: str, message: Any):
        self.comm.send(self.name, to, message, header)
    
    def fetch(self, from_role: str, header: str):
        return self.comm.fetch(self.name, from_role, header)
    
    def fetch_and_store(self, from_role: str, header: str):
        self.storage[header] = self.fetch(from_role, header)
    