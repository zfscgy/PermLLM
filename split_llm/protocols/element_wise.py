from typing import Callable

import torch

from split_llm.common.communication import Communication, Node, SimulatedCommunication
from split_llm.protocols.base_protocols import SS_Perm
from split_llm.protocols.base import Protocol

from split_llm.common.utils import test_func


class SS_ElementWise__RandPerm(Protocol):
    def __init__(self, x_shape, f_perm: Callable, f_invperm: Callable, f_elemwise: Callable,
                name: str, 
                node_0: Node, node_1: Node, node_2: Node,
                mask_scale: float, device: str="cpu") -> None:
        """
        f_perm(x, perm)
        """
        self.x_shape = x_shape
        self.f_perm = f_perm
        self.f_invperm = f_invperm
        self.f_elemwise = f_elemwise
        self.name = name
        self.node_0 = node_0
        self.node_1 = node_1
        self.node_2 = node_2
        self.mask_scale = mask_scale
        self.device = device


        self.perm_name = f"{self.name}/perm"
        self.perm_protocol = SS_Perm(x_shape, f_perm, self.perm_name, node_0, node_1, node_2, mask_scale, device)

        self.invperm_name = f"{self.name}/invperm"
        self.invperm_protocol = SS_Perm(x_shape, f_perm, self.invperm_name, node_0, node_1, node_2, mask_scale, device)

    
    def prepare(self):
        self.perm_protocol.prepare()
        self.invperm_protocol.prepare()

    def offline_execute(self):
        """
        Input:
            Node 0: new_perm, new_invperm
            Those two permutations must be a pair, and the latter is the inverse of the former,
            i.e., f_invperm[f_perm(X, new_perm), new_invperm] = X
        """
        self.node_0.storage[f"{self.perm_name}:new_perm"] = self.node_0.storage[f"{self.name}:new_perm"]
        self.node_0.storage[f"{self.invperm_name}:new_perm"] = self.node_0.storage[f"{self.name}:new_invperm"]
        self.perm_protocol.offline_execute()
        self.invperm_protocol.offline_execute()

    def online_execute(self):
        """
        Input:
            Node 0: x0
            Node 1: x1
        """
        
        # In node_0
        self.node_0.storage[f"{self.perm_name}:x0"] = self.node_0.storage[f"{self.name}:x0"]
        # In node_1
        self.node_1.storage[f"{self.perm_name}:x1"] = self.node_1.storage[f"{self.name}:x1"]


        # Subprotocol
        self.perm_protocol.online_execute()

        # In node_0
        self.node_0.send(self.node_1.name, f"{self.name}:permuted-x0", self.node_0.storage[f"{self.perm_name}:z0"])

        # In node_1
        permuted_x = self.node_1.fetch(self.node_0.name, f"{self.name}:permuted-x0") + self.node_1.storage[f"{self.perm_name}:z1"]
        permuted_y = self.f_elemwise(permuted_x)

        self.node_1.storage[f"{self.invperm_name}:x1"] = permuted_y

        y_shape = permuted_y.shape
        del permuted_x, permuted_y



        # In node_0
        self.node_0.storage[f"{self.invperm_name}:x0"] = torch.zeros(*y_shape, device=self.device, dtype=torch.float)

        # Subprotocol
        self.invperm_protocol.online_execute()

        # In node_0
        self.node_0.storage[f"{self.name}:z0"] = self.node_0.storage[f"{self.invperm_name}:z0"]
        # In node_1
        self.node_1.storage[f"{self.name}:z1"] = self.node_1.storage[f"{self.invperm_name}:z1"]

        self.perm_protocol.clear_io()
        self.invperm_protocol.clear_io()

    def clear_io(self):
        del self.node_0.storage[f"{self.name}:x0"], self.node_0.storage[f"{self.name}:z0"]
        del self.node_1.storage[f"{self.name}:x1"], self.node_1.storage[f"{self.name}:z1"]



if __name__ == "__main__":

    @test_func
    def test__SS_ElementWise__RandPerm():
        communication = SimulatedCommunication(["n0", "n1", "n2"])
        communication.new_stage("Test")

        n0 = Node(communication, "n0")
        n1 = Node(communication, "n1")
        n2 = Node(communication, "n2")

        protocol_name = "ss_elementwise"
        protocol = SS_ElementWise__RandPerm(
            [4], (lambda x, p: x[p]), (lambda x, p: x[p]), (lambda x: x**3),
            protocol_name, n0, n1, n2, 10)
        x = torch.tensor([1, 9, 2, 6]).float()
        x0 = torch.rand_like(x) * 10 - 5

        n0.storage[f"{protocol_name}:new_perm"] = torch.tensor([3, 2, 1, 0])
        n0.storage[f"{protocol_name}:new_invperm"] = torch.tensor([3, 2, 1, 0])

        protocol.prepare()
        protocol.offline_execute()

        n0.storage[f"{protocol_name}:x0"] = x0
        n1.storage[f"{protocol_name}:x1"] = x - x0
        protocol.online_execute()

        print(n0.storage[f"{protocol_name}:z0"] + n1.storage[f"{protocol_name}:z1"])

    test__SS_ElementWise__RandPerm()