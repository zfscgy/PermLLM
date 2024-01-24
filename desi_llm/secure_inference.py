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

class NetworkSimulator:
    def __init__(self, latency: float, speed: float):
        """
        latency: time lag
        speed: number of **bytes** sent per minute
        """
        self.latency = latency
        self.speed = speed
    
        self.total_time: int = 0
        self.total_comm: float = 0

        self.all_history = []
    
    def transfer(self, m, desc: str = ""):
        m_size = estimate_size(m)
        self.total_comm += m_size

        m_time = self.latency + (self.total_comm) / self.speed
        self.total_time += m_time
        self.all_history.append((m_size, m_time, desc))
