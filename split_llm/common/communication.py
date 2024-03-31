from typing import Any, Tuple, List

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
    


class SimulatedCommunication:
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

    def send(self, from_role: str, to_role: str, message: Any, header: str, stage: int = None, end_stage: int = None):
        msg_size = estimate_size(message)
        self.communication_buffer[header] = message
        self.comm_history[self.current_stage].append({
            "from": from_role,
            "to": to_role,
            "header": header,
            "size": msg_size,
            "end": end_stage
        })

    def generate_report(self, latency: float, bandwidth: float):
        """
        The bandwidth's unit is bytes.
        """
        comms = [0 for _ in range(self.current_stage)]
        times = [0 for _ in range(self.current_stage)]
        for s in range(self.current_stage):
            pass