from typing import Callable
import torch
from split_llm.common.communication import Communication, Node


class SS_Mul:
    def __init__(self, x_shape, y_shape, z_shape, f_mul: Callable, 
                 name: str, 
                 node_0: Node, node_1: Node, node_2: Node,
                 mask_scale: float, device: str="cuda") -> None:
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.z_shape = z_shape
        self.f_mul = f_mul
        self.name = name
        self.node_0 = node_0
        self.node_1 = node_1
        self.node_2 = node_2
        self.mask_scale = mask_scale
        self.device = device

    def prepare_x(self, x_0: torch.Tensor, x_1: torch.Tensor):
        """
        We assume that X is constant
        """
        
        # In node_2
        u = torch.rand(*self.x_shape, device=self.device) * self.mask_scale - 0.5 * self.mask_scale
