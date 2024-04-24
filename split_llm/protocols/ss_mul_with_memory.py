from typing import Callable

import torch

from split_llm.common.communication import Communication, Node, SimulatedCommunication
from split_llm.common.utils import test_func

from split_llm.protocols.base import Protocol


class SS_Mul__AppendingX(Protocol):
    def __init__(self, x_shape, appending_dim: int, f_mul: Callable,
                name: str, 
                node_0: Node, node_1: Node, node_2: Node,
                mask_scale: float, device: str="cpu") -> None:
        """
        X's size increases every time
        e.g., 
        """
        self.x_shape = x_shape
        self.appending_dim = appending_dim
        self.f_mul = f_mul
        self.name = name
        self.node_0 = node_0
        self.node_1 = node_1
        self.node_2 = node_2
        self.mask_scale = mask_scale
        self.device = device
    
        self.appended_size_offline = 0
        self.appended_size_online = 0

    def prepare(self):
        # In node_2
        # Prepare beaver_u with max length
        u0 = torch.rand(*self.x_shape, device=self.device) * self.mask_scale - 0.5 * self.mask_scale
        u1 = torch.rand(*self.x_shape, device=self.device) * self.mask_scale - 0.5 * self.mask_scale
        u = u0 + u1
        self.node_2.storage[f"{self.name}:beaver_u_extended"] = u
        self.node_2.send(self.node_0.name, f"{self.name}:beaver_u0 extended", u0)
        self.node_2.send(self.node_1.name, f"{self.name}:beaver_u1 extended", u1)
        self.node_2.storage[f"{self.name}:appended_size"] = 0

        del u0, u1, u

        # In node_0
        self.node_0.fetch_and_store(self.node_2.name, f"{self.name}:beaver_u0 extended")

        # In node_1        
        self.node_1.fetch_and_store(self.node_2.name, f"{self.name}:beaver_u1 extended")



    def offline_execute(self, y_shape, z_shape, append_size: int):
        """
        append_size: the size of the tensor appendded to x (in appending_dim)
        """
        self.appended_size_offline += append_size
        u = torch.index_select(self.node_2.storage[f"{self.name}:beaver_u_extended"], self.appending_dim, torch.arange(0, self.appended_size_offline, device=self.device))

        v0 = torch.rand(*y_shape, device=self.device) * self.mask_scale - 0.5 * self.mask_scale
        v1 = torch.rand(*y_shape, device=self.device) * self.mask_scale - 0.5 * self.mask_scale
        v = v0 + v1


        w = self.f_mul(u, v)
        w0 = torch.rand(z_shape, device=self.device) * self.mask_scale ** 2 - 0.5 * self.mask_scale ** 2
        w1 = w - w0

        self.node_2.send(self.node_0.name, f"{self.name}:beaver_v0, w0", [v0, w0])

        self.node_2.send(self.node_1.name, f"{self.name}:beaver_v1, w1", [v1, w1])

        del v0, v1, v, w0, w1, w


        # In node_0
        self.node_0.fetch_and_enqueue(self.node_2.name, f"{self.name}:beaver_v0, w0")

        # In node_1
        self.node_1.fetch_and_enqueue(self.node_2.name, f"{self.name}:beaver_v1, w1")


    def online_execute(self):
        """
        Output:
        Node 0 holds z0
        Node 1 holds z1
        z0 + z1 = x * y
        """
        appended_size = self.node_0.storage[f"{self.name}:x0 appended"].shape[self.appending_dim]
        self.appended_size_online += appended_size

        # In node_0
        x0_sub_u0_appended = self.node_0.storage[f"{self.name}:x0 appended"] - \
            torch.index_select(self.node_0.storage[f"{self.name}:beaver_u0 extended"], self.appending_dim, 
                         torch.arange(self.appended_size_online - appended_size, self.appended_size_online, device=self.device))
        y0_sub_v0 = self.node_0.storage[f"{self.name}:y0"] - self.node_0.storage[f"{self.name}:beaver_v0, w0"][-1][0]
    
        self.node_0.storage[f"{self.name}:x0-u0 appended"] = x0_sub_u0_appended
        self.node_0.storage[f"{self.name}:y0-v0"] = y0_sub_v0
        
        self.node_0.send(self.node_1.name, f"{self.name}:x0-u0 appended, y0-v0", [x0_sub_u0_appended, y0_sub_v0])

        del x0_sub_u0_appended, y0_sub_v0

        # In node_1

        x1_sub_u1_appended = self.node_1.storage[f"{self.name}:x1 appended"] - \
            torch.index_select(self.node_1.storage[f"{self.name}:beaver_u1 extended"], self.appending_dim, 
                         torch.arange(self.appended_size_online - appended_size, self.appended_size_online, device=self.device))
        y1_sub_v1 = self.node_1.storage[f"{self.name}:y1"] - self.node_1.storage[f"{self.name}:beaver_v1, w1"][-1][0]

        self.node_1.storage[f"{self.name}:y1-v1"] = y1_sub_v1

        self.node_1.send(self.node_0.name, f"{self.name}:x1-u1 appended, y1-v1", [x1_sub_u1_appended, y1_sub_v1])


        x0_sub_u0_appended, y0_sub_v0 = self.node_1.fetch(self.node_0.name, f"{self.name}:x0-u0 appended, y0-v0")
        if f"{self.name}:x-u" not in self.node_0.storage:
            self.node_1.storage[f"{self.name}:x-u"] = x1_sub_u1_appended + x0_sub_u0_appended
        else:
            self.node_1.storage[f"{self.name}:x-u"] = torch.cat([
                self.node_1.storage[f"{self.name}:x-u"], 
                x1_sub_u1_appended + x0_sub_u0_appended], 
                dim=self.appending_dim)
        y_sub_v = y0_sub_v0 + self.node_1.storage[f"{self.name}:y1-v1"]

        u1 = torch.index_select(self.node_1.storage[f"{self.name}:beaver_u1 extended"], self.appending_dim, torch.arange(self.appended_size_online, device=self.device))
        v1, w1 = self.node_1.storage[f"{self.name}:beaver_v1, w1"].pop()
        self.node_1.storage[f"{self.name}:z1"] = self.f_mul(u1, y_sub_v) + self.f_mul(self.node_1.storage[f"{self.name}:x-u"], v1) + w1
        del x1_sub_u1_appended, y1_sub_v1, x0_sub_u0_appended, y0_sub_v0, y_sub_v, u1, v1, w1

        # In node_0
        x1_sub_u1_appended, y1_sub_v1 = self.node_0.fetch(self.node_1.name, f"{self.name}:x1-u1 appended, y1-v1")
        if f"{self.name}:x-u" not in self.node_0.storage:
            self.node_0.storage[f"{self.name}:x-u"] = \
                self.node_0.storage[f"{self.name}:x0-u0 appended"] + x1_sub_u1_appended
        else:
            self.node_0.storage[f"{self.name}:x-u"] = torch.cat([
                self.node_0.storage[f"{self.name}:x-u"], 
                self.node_0.storage[f"{self.name}:x0-u0 appended"] + x1_sub_u1_appended], 
                dim=self.appending_dim)
        
        y_sub_v = self.node_0.storage[f"{self.name}:y0-v0"] + y1_sub_v1

        u0 = torch.index_select(self.node_0.storage[f"{self.name}:beaver_u0 extended"], self.appending_dim, torch.arange(self.appended_size_online, device=self.device))
        v0, w0 = self.node_0.storage[f"{self.name}:beaver_v0, w0"].pop()
        self.node_0.storage[f"{self.name}:z0"] = \
              self.f_mul(self.node_0.storage[f"{self.name}:x-u"], y_sub_v) + \
              self.f_mul(u0, y_sub_v) + self.f_mul(self.node_0.storage[f"{self.name}:x-u"], v0) + w0

        del x1_sub_u1_appended, y_sub_v, u0, v0, w0, self.node_0.storage[f"{self.name}:x0-u0 appended"]


        # Clear cache
        del self.node_0.storage[f"{self.name}:y0-v0"]
        del self.node_1.storage[f"{self.name}:y1-v1"]

    def clear_io(self):
        del self.node_0.storage[f"{self.name}:x0 appended"], self.node_0.storage[f"{self.name}:y0"], self.node_0.storage[f"{self.name}:z0"]
        del self.node_1.storage[f"{self.name}:x1 appended"], self.node_1.storage[f"{self.name}:y1"], self.node_1.storage[f"{self.name}:z1"]

if __name__ == "__main__":
    def test__SS_Mul__AppendingX():
        communication = SimulatedCommunication(["n0", "n1", "n2"])
        communication.new_stage("Test")

        n0 = Node(communication, "n0")
        n1 = Node(communication, "n1")
        n2 = Node(communication, "n2")

        protocol_name = "ss_mul__cx_n0"

        protocol = SS_Mul__AppendingX([1, 10], 1, torch.matmul, protocol_name, n0, n1, n2, 10)
        protocol.prepare()

        protocol.offline_execute([5, 1], [1, 1], 5)
        protocol.offline_execute([8, 1], [1, 1], 3)
        protocol.offline_execute([10, 1], [1, 1], 2)
        

        # 1
        x_appended = torch.tensor([[1, 2, 3, 4, 5]]).float()
        x0_appended = torch.rand_like(x_appended) * 5 - 2.5
        x1_appended = x_appended - x0_appended
        n0.storage[f"{protocol_name}:x0 appended"] = x0_appended
        n1.storage[f"{protocol_name}:x1 appended"] = x1_appended
        
        y = torch.ones([5, 1])
        y0 = torch.rand_like(y) * 5 - 2.5
        y1 = y - y0
        n0.storage[f"{protocol_name}:y0"] = y0
        n1.storage[f"{protocol_name}:y1"] = y1

        protocol.online_execute()
        print(n0.storage[f"{protocol_name}:z0"] + n1.storage[f"{protocol_name}:z1"])


        # 2
        x_appended = torch.tensor([[6, 7, 8]]).float()
        x0_appended = torch.rand_like(x_appended) * 5 - 2.5
        x1_appended = x_appended - x0_appended
        n0.storage[f"{protocol_name}:x0 appended"] = x0_appended
        n1.storage[f"{protocol_name}:x1 appended"] = x1_appended
        
        y = torch.ones([8, 1])
        y0 = torch.rand_like(y) * 5 - 2.5
        y1 = y - y0
        n0.storage[f"{protocol_name}:y0"] = y0
        n1.storage[f"{protocol_name}:y1"] = y1

        protocol.online_execute()
        print(n0.storage[f"{protocol_name}:z0"] + n1.storage[f"{protocol_name}:z1"])


        # 3
        x_appended = torch.tensor([[9, 10]]).float()
        x0_appended = torch.rand_like(x_appended) * 5 - 2.5
        x1_appended = x_appended - x0_appended
        n0.storage[f"{protocol_name}:x0 appended"] = x0_appended
        n1.storage[f"{protocol_name}:x1 appended"] = x1_appended
        
        y = torch.ones([10, 1])
        y0 = torch.rand_like(y) * 5 - 2.5
        y1 = y - y0
        n0.storage[f"{protocol_name}:y0"] = y0
        n1.storage[f"{protocol_name}:y1"] = y1

        protocol.online_execute()
        print(n0.storage[f"{protocol_name}:z0"] + n1.storage[f"{protocol_name}:z1"])


        print("----------Storage----------")
        print(n0.storage)
        print(n1.storage)    
        print("----------Storage (after clear IO)-----------")
        protocol.clear_io()
        print(n0.storage)
        print(n1.storage)  


    test__SS_Mul__AppendingX()