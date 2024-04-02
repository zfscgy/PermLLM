from ast import main
from typing import Callable
import torch
from split_llm.common.communication import Communication, Node, SimulatedCommunication


class SS_Mul__CX_N0_Y_N1:
    def __init__(self, x_shape, y_shape, z_shape, f_mul: Callable, 
                 name: str, 
                 node_0: Node, node_1: Node, node_2: Node,
                 mask_scale: float, device: str="cpu") -> None:
        """
        X is constant in node_0 (name:x)
        Y is not constant in node_1 (name:y)
        """
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

    def prepare(self):
        """
        We assume that X is constant and is obtained by node_0
        """
        
        # In node_2
        u = torch.rand(*self.x_shape, device=self.device) * self.mask_scale - 0.5 * self.mask_scale
        # Prepare the shares of U
        self.node_2.storage[f"{self.name}:beaver_u"] = u

        self.node_2.send(self.node_0.name, f"{self.name}:beaver_u", u)

        del u

        # In node_0
        u = self.node_0.fetch(self.node_2.name, f"{self.name}:beaver_u")
        x_sub_u = self.node_0.storage[f"{self.name}:x"] - u
        self.node_0.send(self.node_1.name, f"{self.name}:x-u", x_sub_u)
        
        del u, x_sub_u

        # In node_1
        self.node_1.fetch_and_store(self.node_0.name, f"{self.name}:x-u")


    def offline_execute(self):
        # In node_2
        u = self.node_2.storage[f"{self.name}:beaver_u"]

        # Prepare the beaver triples
        v = torch.rand(*self.y_shape, device=self.device) * self.mask_scale - 0.5 * self.mask_scale
        self.node_2.send(self.node_1.name, f"{self.name}:beaver_v", v)


        w = self.f_mul(u, v)
        w0 = torch.rand(*self.z_shape, device=self.device) * self.mask_scale ** 2 - 0.5 * self.mask_scale ** 2
        w1 = w - w0

        self.node_2.send(self.node_0.name, f"{self.name}:beaver_w0", w0)
        self.node_2.send(self.node_1.name, f"{self.name}:beaver_w1", w1)

        del u, v, w0, w1, w

        # In node_1
        self.node_1.fetch_and_store(self.node_2.name, f"{self.name}:beaver_v")
        self.node_1.fetch_and_store(self.node_2.name, f"{self.name}:beaver_w1")
        self.node_1.storage[f"{self.name}:z1"] = self.f_mul(
            self.node_1.storage[f"{self.name}:x-u"], 
            self.node_1.storage[f"{self.name}:beaver_v"]) + \
            self.node_1.storage[f"{self.name}:beaver_w1"]


        # In node_0
        self.node_0.fetch_and_store(self.node_2.name, f"{self.name}:beaver_w0")


    def online_execute(self):
        # In node_1
        y_sub_v = self.node_1.storage[f"{self.name}:y"] - self.node_1.storage[f"{self.name}:beaver_v"]
        self.node_1.send(self.node_0.name, f"{self.name}:y-v", y_sub_v)
        del y_sub_v
    
        # In node_0
        y_sub_v = self.node_0.fetch(self.node_1.name, f"{self.name}:y-v")
        z0 = self.f_mul(self.node_0.storage[f"{self.name}:x"], y_sub_v) + self.node_0.storage[f"{self.name}:beaver_w0"]
        self.node_0.storage[f"{self.name}:z0"] = z0

        del self.node_0.storage[f"{self.name}:beaver_w0"], self.node_1.storage[f"{self.name}:beaver_v"], self.node_1.storage[f"{self.name}:beaver_w1"]
        del self.node_1.storage[f"{self.name}:y"]


class SS_Mul:
    def __init__(self, x_shape, y_shape, z_shape, f_mul: Callable, 
                 name: str, 
                 node_0: Node, node_1: Node, node_2: Node,
                 mask_scale: float, device: str="cpu") -> None:
        """
        Node_0 holds name:x0, name:y0
        Node_1 holds name:x1, name:y1
        """
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

    def prepare(self):
        pass

    def offline_execute(self):
        # In node_2
        # Prepare beaver triples

        u0 = torch.rand(*self.x_shape, device=self.device) * self.mask_scale - 0.5 * self.mask_scale
        u1 = torch.rand(*self.x_shape, device=self.device) * self.mask_scale - 0.5 * self.mask_scale
        u = u0 + u1

        v0 = torch.rand(*self.y_shape, device=self.device) * self.mask_scale - 0.5 * self.mask_scale
        v1 = torch.rand(*self.y_shape, device=self.device) * self.mask_scale - 0.5 * self.mask_scale
        v = v0 + v1


        w = self.f_mul(u, v)
        w0 = torch.rand(*self.z_shape, device=self.device) * self.mask_scale ** 2 - 0.5 * self.mask_scale ** 2
        w1 = w - w0


        self.node_2.send(self.node_0.name, f"{self.name}:beaver_u0", u0)
        self.node_2.send(self.node_1.name, f"{self.name}:beaver_u1", u1)

        self.node_2.send(self.node_0.name, f"{self.name}:beaver_v0", v0)
        self.node_2.send(self.node_1.name, f"{self.name}:beaver_v1", v1)


        self.node_2.send(self.node_0.name, f"{self.name}:beaver_w0", w0)
        self.node_2.send(self.node_1.name, f"{self.name}:beaver_w1", w1)

        del v0, v1, v, u0, u1, u, w0, w1, w


        # In node_0
        self.node_0.fetch_and_store(self.node_2.name, f"{self.name}:beaver_u0")
        self.node_0.fetch_and_store(self.node_2.name, f"{self.name}:beaver_v0")
        self.node_0.fetch_and_store(self.node_2.name, f"{self.name}:beaver_w0")

        # In node_1
        self.node_1.fetch_and_store(self.node_2.name, f"{self.name}:beaver_u1")
        self.node_1.fetch_and_store(self.node_2.name, f"{self.name}:beaver_v1")
        self.node_1.fetch_and_store(self.node_2.name, f"{self.name}:beaver_w1")


    def online_execute(self):
        # In node_0
        x0_sub_u0 = self.node_0.storage[f"{self.name}:x0"] - self.node_0.storage[f"{self.name}:beaver_u0"]
        y0_sub_v0 = self.node_0.storage[f"{self.name}:y0"] - self.node_0.storage[f"{self.name}:beaver_v0"]

        self.node_0.storage[f"{self.name}:x0-u0"] = x0_sub_u0
        self.node_0.storage[f"{self.name}:y0-v0"] = y0_sub_v0
        
        self.node_0.send(self.node_1.name, f"{self.name}:x0-u0", x0_sub_u0)
        self.node_0.send(self.node_1.name, f"{self.name}:y0-v0", y0_sub_v0)

        del x0_sub_u0, y0_sub_v0

        # In node_1
        x1_sub_u1 = self.node_1.storage[f"{self.name}:x1"] - self.node_1.storage[f"{self.name}:beaver_u1"]
        y1_sub_v1 = self.node_1.storage[f"{self.name}:y1"] - self.node_1.storage[f"{self.name}:beaver_v1"]

        self.node_1.storage[f"{self.name}:x1-u1"] = x1_sub_u1
        self.node_1.storage[f"{self.name}:y1-v1"] = y1_sub_v1
        
        self.node_1.send(self.node_0.name, f"{self.name}:x1-u1", x1_sub_u1)
        self.node_1.send(self.node_0.name, f"{self.name}:y1-v1", y1_sub_v1)

        # In node_0
        x_sub_u = self.node_0.storage[f"{self.name}:x0-u0"] + self.node_0.fetch(self.node_1.name, f"{self.name}:x1-u1")
        y_sub_v = self.node_0.storage[f"{self.name}:y0-v0"] + self.node_0.fetch(self.node_1.name, f"{self.name}:y1-v1")

        self.node_0.storage[f"{self.name}:z0"] = self.f_mul(x_sub_u, y_sub_v) + \
              self.f_mul(self.node_0.storage[f"{self.name}:beaver_u0"], y_sub_v) + \
              self.f_mul(x_sub_u, self.node_0.storage[f"{self.name}:beaver_v0"]) + \
              self.node_0.storage[f"{self.name}:beaver_w0"]

        del x_sub_u, y_sub_v

        # In node_1
        x_sub_u = self.node_1.fetch(self.node_0.name, f"{self.name}:x0-u0") + self.node_1.storage[f"{self.name}:x1-u1"]
        y_sub_v = self.node_1.fetch(self.node_0.name, f"{self.name}:y0-v0") + self.node_1.storage[f"{self.name}:y1-v1"]

        self.node_1.storage[f"{self.name}:z1"] = \
              self.f_mul(self.node_1.storage[f"{self.name}:beaver_u1"], y_sub_v) + \
              self.f_mul(x_sub_u, self.node_1.storage[f"{self.name}:beaver_v1"]) + \
              self.node_1.storage[f"{self.name}:beaver_w1"]
        
        del x_sub_u, y_sub_v


        del self.node_0.storage[f"{self.name}:x0"], self.node_0.storage[f"{self.name}:y0"]
        del self.node_0.storage[f"{self.name}:beaver_u0"], self.node_0.storage[f"{self.name}:beaver_v0"], self.node_0.storage[f"{self.name}:beaver_w0"]
        del self.node_0.storage[f"{self.name}:x0-u0"], self.node_0.storage[f"{self.name}:y0-v0"]

        del self.node_1.storage[f"{self.name}:x1"], self.node_1.storage[f"{self.name}:y1"]
        del self.node_1.storage[f"{self.name}:beaver_u1"], self.node_1.storage[f"{self.name}:beaver_v1"], self.node_1.storage[f"{self.name}:beaver_w1"]
        del self.node_1.storage[f"{self.name}:x1-u1"], self.node_1.storage[f"{self.name}:y1-v1"]



if __name__ == "__main__":
    def test__SS_Mul__CX_N0_Y_N1():
        # Test SS_Mul__CX_N0_Y_N1
        communication = SimulatedCommunication(["n0", "n1", "n2"])
        communication.new_stage("Test")

        n0 = Node(communication, "n0")
        n1 = Node(communication, "n1")
        n2 = Node(communication, "n2")

        protocol_name = "ss_mul__cx_n0_y_n1"
        n0.storage[f"{protocol_name}:x"] = torch.tensor([[1, 2]]).float()
        n1.storage[f"{protocol_name}:y"] = torch.tensor([[2], [1]]).float()

        protocol = SS_Mul__CX_N0_Y_N1([1, 2], [2, 1], [1, 1], torch.matmul, protocol_name, n0, n1, n2, 10)
        protocol.prepare()
        protocol.offline_execute()
        protocol.online_execute()

        print(n0.storage)
        print(n1.storage)

    def test__SS_Mul():
        # Test SS_Mul__CX_N0_Y_N1
        communication = SimulatedCommunication(["n0", "n1", "n2"])
        communication.new_stage("Test")

        n0 = Node(communication, "n0")
        n1 = Node(communication, "n1")
        n2 = Node(communication, "n2")

        protocol_name = "ss_mul"
        n0.storage[f"{protocol_name}:x0"] = torch.tensor([[1, 2]]).float()
        n0.storage[f"{protocol_name}:y0"] = torch.tensor([[0], [0]]).float()
        
        n1.storage[f"{protocol_name}:y1"] = torch.tensor([[2], [1]]).float()
        n1.storage[f"{protocol_name}:x1"] = torch.tensor([[0, 0]]).float()
        

        protocol = SS_Mul([1, 2], [2, 1], [1, 1], torch.matmul, protocol_name, n0, n1, n2, 10)
        protocol.offline_execute()
        protocol.online_execute()

        print(n0.storage)
        print(n1.storage)


    test__SS_Mul__CX_N0_Y_N1()
    test__SS_Mul()