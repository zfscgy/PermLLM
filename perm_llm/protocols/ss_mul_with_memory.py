from typing import Callable

import torch

from perm_llm.common.communication import Communication, Node, SimulatedCommunication
from perm_llm.common.utils import test_func

from perm_llm.protocols.base import Protocol


class SS_Mul__AppendingX(Protocol):
    def __init__(self, appending_dim: int, f_mul: Callable,
                name: str, 
                node_0: Node, node_1: Node, node_2: Node,
                mask_scale: float, device: str="cpu") -> None:
        """
        X's size increases every time
        e.g., 
        """
        self.appending_dim = appending_dim
        self.f_mul = f_mul
        self.name = name
        self.node_0 = node_0
        self.node_1 = node_1
        self.node_2 = node_2

        if not isinstance(mask_scale, dict):
            mask_scale = {
                "u": mask_scale,
                "v": mask_scale,
                "w": mask_scale
            }
        self.mask_scale = mask_scale
        self.device = device

    def prepare(self):
        pass

    def offline_execute(self, x_appended_shape, y_shape, z_shape):
        # In node_2
        if self.node_2.local():
            # Prepare beaver_u with max length
            u0_appended = torch.rand(*x_appended_shape, device=self.device) * self.mask_scale['u'] - 0.5 * self.mask_scale['u']
            u1_appended = torch.rand(*x_appended_shape, device=self.device) * self.mask_scale['u'] - 0.5 * self.mask_scale['u']
            u_appended = u0_appended + u1_appended
            if f"{self.name}:beaver_u" not in self.node_2.storage: 
                self.node_2.storage[f"{self.name}:beaver_u"] = u_appended
            else:
                self.node_2.storage[f"{self.name}:beaver_u"] = torch.cat([self.node_2.storage[f"{self.name}:beaver_u"], u_appended], dim=self.appending_dim)
            
            u = self.node_2.storage[f"{self.name}:beaver_u"]
            v0 = torch.rand(*y_shape, device=self.device) * self.mask_scale['v'] - 0.5 * self.mask_scale['v']
            v1 = torch.rand(*y_shape, device=self.device) * self.mask_scale['v'] - 0.5 * self.mask_scale['v']
            v = v0 + v1
            w = self.f_mul(u, v)
            w0 = torch.rand(z_shape, device=self.device) * self.mask_scale['w'] - 0.5 * self.mask_scale['w']
            w1 = w - w0
            
            self.node_2.send(self.node_0.name, f"{self.name}:beaver_u0 appended, v0, w0", [u0_appended, v0, w0])
            self.node_2.send(self.node_1.name, f"{self.name}:beaver_u1 appended, v1, w1", [u1_appended, v1, w1])

            del u0_appended, u1_appended, u_appended, u, v0, v1, v, w0, w1, w
        


        # In node_0
        if self.node_0.local():
            self.node_0.fetch_and_enqueue(self.node_2.name, f"{self.name}:beaver_u0 appended, v0, w0")

        # In node_1
        if self.node_1.local():        
            self.node_1.fetch_and_enqueue(self.node_2.name, f"{self.name}:beaver_u1 appended, v1, w1")


    def online_execute(self):
        """
        Input:
            Node 0 holds x0 appended
            Node 1 holds x1 appended
        Output:
            Node 0 holds z0
            Node 1 holds z1
        z0 + z1 = x * y
        """

        # In node_0
        if self.node_0.local():
            u0_appended = self.node_0.storage[f"{self.name}:beaver_u0 appended, v0, w0"][-1][0]
            if f"{self.name}:beaver_u0" not in self.node_0.storage:
                self.node_0.storage[f"{self.name}:beaver_u0"] = u0_appended
            else:
                self.node_0.storage[f"{self.name}:beaver_u0"] = torch.cat([self.node_0.storage[f"{self.name}:beaver_u0"], u0_appended], dim=self.appending_dim)

            x0_sub_u0_appended = self.node_0.storage[f"{self.name}:x0 appended"] - u0_appended
            y0_sub_v0 = self.node_0.storage[f"{self.name}:y0"] - self.node_0.storage[f"{self.name}:beaver_u0 appended, v0, w0"][-1][1]
        
            self.node_0.storage[f"{self.name}:x0-u0 appended"] = x0_sub_u0_appended
            self.node_0.storage[f"{self.name}:y0-v0"] = y0_sub_v0
            
            self.node_0.send(self.node_1.name, f"{self.name}:x0-u0 appended, y0-v0", [x0_sub_u0_appended, y0_sub_v0])

            del u0_appended, x0_sub_u0_appended, y0_sub_v0

        # In node_1
        if self.node_1.local():
            u1_appended = self.node_1.storage[f"{self.name}:beaver_u1 appended, v1, w1"][-1][0]
            if f"{self.name}:beaver_u1" not in self.node_1.storage:
                self.node_1.storage[f"{self.name}:beaver_u1"] = u1_appended
            else:
                self.node_1.storage[f"{self.name}:beaver_u1"] = torch.cat([self.node_1.storage[f"{self.name}:beaver_u1"], u1_appended], dim=self.appending_dim)

            x1_sub_u1_appended = self.node_1.storage[f"{self.name}:x1 appended"] - u1_appended
            y1_sub_v1 = self.node_1.storage[f"{self.name}:y1"] - self.node_1.storage[f"{self.name}:beaver_u1 appended, v1, w1"][-1][1]

            self.node_1.send(self.node_0.name, f"{self.name}:x1-u1 appended, y1-v1", [x1_sub_u1_appended, y1_sub_v1])

            x0_sub_u0_appended, y0_sub_v0 = self.node_1.fetch(self.node_0.name, f"{self.name}:x0-u0 appended, y0-v0")
            if f"{self.name}:x-u" not in self.node_1.storage:
                self.node_1.storage[f"{self.name}:x-u"] = x1_sub_u1_appended + x0_sub_u0_appended
            else:
                self.node_1.storage[f"{self.name}:x-u"] = torch.cat([
                    self.node_1.storage[f"{self.name}:x-u"], 
                    x1_sub_u1_appended + x0_sub_u0_appended], 
                    dim=self.appending_dim)

            y_sub_v = y0_sub_v0 + y1_sub_v1

            u1 = self.node_1.storage[f"{self.name}:beaver_u1"]
            v1, w1 = self.node_1.storage[f"{self.name}:beaver_u1 appended, v1, w1"].pop()[1:]
            self.node_1.storage[f"{self.name}:z1"] = self.f_mul(u1, y_sub_v) + self.f_mul(self.node_1.storage[f"{self.name}:x-u"], v1) + w1
            del u1_appended, x1_sub_u1_appended, y1_sub_v1, x0_sub_u0_appended, y0_sub_v0, y_sub_v, u1, v1, w1

        # In node_0
        if self.node_0.local():
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

            u0 = self.node_0.storage[f"{self.name}:beaver_u0"]
            v0, w0 = self.node_0.storage[f"{self.name}:beaver_u0 appended, v0, w0"].pop()[1:]
            self.node_0.storage[f"{self.name}:z0"] = \
                self.f_mul(self.node_0.storage[f"{self.name}:x-u"], y_sub_v) + \
                self.f_mul(u0, y_sub_v) + self.f_mul(self.node_0.storage[f"{self.name}:x-u"], v0) + w0

            del x1_sub_u1_appended, y_sub_v, u0, v0, w0, self.node_0.storage[f"{self.name}:x0-u0 appended"]


        # Clear cache
        if self.node_0.local():
            del self.node_0.storage[f"{self.name}:y0-v0"]


    def clear_io(self):
        if self.node_0.local():
            del self.node_0.storage[f"{self.name}:x0 appended"], self.node_0.storage[f"{self.name}:y0"], self.node_0.storage[f"{self.name}:z0"]
        
        if self.node_1.local():
            del self.node_1.storage[f"{self.name}:x1 appended"], self.node_1.storage[f"{self.name}:y1"], self.node_1.storage[f"{self.name}:z1"]

    def reset(self):
        if self.node_0.local():
            self.node_0.storage[f"{self.name}:beaver_u0 appended, v0, w0"].clear()
            del self.node_0.storage[f"{self.name}:beaver_u0"], self.node_0.storage[f"{self.name}:x-u"]
        
        if self.node_1.local():
            self.node_1.storage[f"{self.name}:beaver_u1 appended, v1, w1"].clear()
            del self.node_1.storage[f"{self.name}:beaver_u1"], self.node_1.storage[f"{self.name}:x-u"]

        if self.node_2.local():
            del self.node_2.storage[f"{self.name}:beaver_u"]

if __name__ == "__main__":
    def test__SS_Mul__AppendingX():
        communication = SimulatedCommunication(["n0", "n1", "n2"])
        communication.new_stage("Test")

        n0 = Node(communication, "n0")
        n1 = Node(communication, "n1")
        n2 = Node(communication, "n2")

        protocol_name = "ss_mul__cx_n0"

        protocol = SS_Mul__AppendingX(1, torch.matmul, protocol_name, n0, n1, n2, 10)
        protocol.prepare()

        protocol.offline_execute([1, 5], [5, 1], [1, 1])
        protocol.offline_execute([1, 3], [8, 1], [1, 1])

        

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
        
        y = torch.ones([8, 1]) * 2
        y0 = torch.rand_like(y) * 5 - 2.5
        y1 = y - y0
        n0.storage[f"{protocol_name}:y0"] = y0
        n1.storage[f"{protocol_name}:y1"] = y1

        protocol.online_execute()
        print(n0.storage[f"{protocol_name}:z0"] + n1.storage[f"{protocol_name}:z1"])


        # 3
        protocol.offline_execute([1, 2], [10, 1], [1, 1])


        x_appended = torch.tensor([[9, 10]]).float()
        x0_appended = torch.rand_like(x_appended) * 5 - 2.5
        x1_appended = x_appended - x0_appended
        n0.storage[f"{protocol_name}:x0 appended"] = x0_appended
        n1.storage[f"{protocol_name}:x1 appended"] = x1_appended
        
        y = torch.ones([10, 1]) * 3
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

        # Adding one useless offline
        protocol.offline_execute([1, 3], [13, 1], [1, 1])
        print("===============\n Test reset... \n ==================")
        protocol.reset()

        protocol.offline_execute([1, 5], [5, 1], [1, 1])
        protocol.offline_execute([1, 3], [8, 1], [1, 1])

        
        # 1
        x_appended = torch.tensor([[1, 2, 3, 4, 5]]).float()
        x0_appended = torch.rand_like(x_appended) * 5 - 2.5
        x1_appended = x_appended - x0_appended
        n0.storage[f"{protocol_name}:x0 appended"] = x0_appended
        n1.storage[f"{protocol_name}:x1 appended"] = x1_appended
        
        y = torch.ones([5, 1]) * 3
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
        
        y = torch.ones([8, 1]) * 2
        y0 = torch.rand_like(y) * 5 - 2.5
        y1 = y - y0
        n0.storage[f"{protocol_name}:y0"] = y0
        n1.storage[f"{protocol_name}:y1"] = y1

        protocol.online_execute()
        print(n0.storage[f"{protocol_name}:z0"] + n1.storage[f"{protocol_name}:z1"])


        # 3
        protocol.offline_execute([1, 2], [10, 1], [1, 1])


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