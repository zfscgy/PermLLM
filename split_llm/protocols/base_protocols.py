from typing import Callable, Union, Dict

import torch

from split_llm.common.communication import Communication, Node, SimulatedCommunication
from split_llm.common.utils import test_func

from split_llm.protocols.base import Protocol


class SS_Mul__CX_N0_Y_N1(Protocol):
    mask_scale_keys = ["u", "v", "w"]
    def __init__(self, x_shape, f_mul: Callable, 
                 name: str, 
                 node_0: Node, node_1: Node, node_2: Node,
                 mask_scale: Union[float, Dict[str, float]], device: str="cpu") -> None:
        """
        X is constant in node_0 
        Y is not constant in node_1
        """
        self.x_shape = x_shape
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
        """
        Input:
            Node 0: X
        """
        
        # In node_2
        if self.node_2.local():
            u = torch.rand(*self.x_shape, device=self.device) * self.mask_scale['u'] - 0.5 * self.mask_scale['u']
            # Prepare the shares of U
            self.node_2.storage[f"{self.name}:beaver_u"] = u

            self.node_2.send(self.node_0.name, f"{self.name}:beaver_u", u)

            del u


        # In node_0
        if self.node_0.local():
            u = self.node_0.fetch(self.node_2.name, f"{self.name}:beaver_u")
            x_sub_u = self.node_0.storage[f"{self.name}:x"] - u
            self.node_0.send(self.node_1.name, f"{self.name}:x-u", x_sub_u)
            
            del u, x_sub_u
        
        # In node_1
        if self.node_1.local():
            self.node_1.fetch_and_store(self.node_0.name, f"{self.name}:x-u")


    def offline_execute(self, y_shape, z_shape):
        # In node_2
        if self.node_2.local():
            u = self.node_2.storage[f"{self.name}:beaver_u"]

            # Prepare the beaver triples
            v = torch.rand(y_shape, device=self.device) * self.mask_scale['v'] - 0.5 * self.mask_scale['v']
            self.node_2.send(self.node_1.name, f"{self.name}:beaver_v", v)


            w = self.f_mul(u, v)
            w0 = torch.rand(z_shape, device=self.device) * self.mask_scale['w'] - 0.5 * self.mask_scale['w']
            w1 = w - w0

            self.node_2.send(self.node_0.name, f"{self.name}:beaver_w0", w0)
            self.node_2.send(self.node_1.name, f"{self.name}:beaver_w1", w1)

            del u, v, w0, w1, w

        if self.node_1.local():
            # In node_1
            self.node_1.fetch_and_enqueue(self.node_2.name, f"{self.name}:beaver_v")
            w1 = self.node_1.fetch(self.node_2.name, f"{self.name}:beaver_w1")
            self.node_1.enqueue(f"{self.name}:z1_cache", self.f_mul(
                self.node_1.storage[f"{self.name}:x-u"], 
                self.node_1.storage[f"{self.name}:beaver_v"][-1]) + w1)

        if self.node_0.local():
            # In node_0
            self.node_0.fetch_and_enqueue(self.node_2.name, f"{self.name}:beaver_w0")


    def online_execute(self):
        """
        Input:
            Node 1: y
        """
        # In node_1
        if self.node_1.local():
            y_sub_v = self.node_1.storage[f"{self.name}:y"] - self.node_1.storage[f"{self.name}:beaver_v"].pop()
            self.node_1.send(self.node_0.name, f"{self.name}:y-v", y_sub_v)
            self.node_1.storage[f"{self.name}:z1"] = self.node_1.storage[f"{self.name}:z1_cache"].pop()
            del y_sub_v

        # In node_0
        if self.node_0.local():
            y_sub_v = self.node_0.fetch(self.node_1.name, f"{self.name}:y-v")
            z0 = self.f_mul(self.node_0.storage[f"{self.name}:x"], y_sub_v) + self.node_0.storage[f"{self.name}:beaver_w0"].pop()
            self.node_0.storage[f"{self.name}:z0"] = z0


    def clear_io(self):
        if self.node_0.local():
            del self.node_0.storage[f"{self.name}:z0"]
        
        if self.node_1.local():
            del self.node_1.storage[f"{self.name}:z1"], self.node_1.storage[f"{self.name}:y"]



class SS_Mul__CX_N0(Protocol):
    mask_scale_keys = ["u", "v", "w"]
    def __init__(self, x_shape, f_mul: Callable, 
                name: str, 
                node_0: Node, node_1: Node, node_2: Node,
                mask_scale: float, device: str="cpu") -> None:
        """
        X is constant in Node 0
        Y is shared
        """
        self.x_shape = x_shape
        self.f_mul = f_mul
        self.name = name
        self.node_0 = node_0
        self.node_1 = node_1
        self.node_2 = node_2

        self.mask_scale = mask_scale
        if not isinstance(mask_scale, dict):
            mask_scale = {
                "u": mask_scale,
                "v": mask_scale,
                "w": mask_scale
            }

        self.device = device
        sub_protocol_name = name + "/SS_Mul__CX_N0_Y_N1"
        self.sub_protocol = SS_Mul__CX_N0_Y_N1(
            x_shape, 
            f_mul, sub_protocol_name, 
            node_0, node_1, node_2,
            mask_scale, device
        )

    def prepare(self):
        """
        Input:
            Node 0: x
        """
        if self.node_0.local():
            self.node_0.storage[f"{self.sub_protocol.name}:x"] = self.node_0.storage[f"{self.name}:x"]
            del self.node_0.storage[f"{self.name}:x"]

        self.sub_protocol.prepare()

    def offline_execute(self, y_shape, z_shape):
        self.sub_protocol.offline_execute(y_shape, z_shape)
    
    def online_execute(self):
        """
        Input:
            Node 0: y0
            Node 1: y1
        """
        # In node_1
        if self.node_1.local():
            self.node_1.storage[f"{self.sub_protocol.name}:y"] = self.node_1.storage[f"{self.name}:y1"]
            del self.node_1.storage[f"{self.name}:y1"]
            
        self.sub_protocol.online_execute()

        # In node_0
        if self.node_0.local():
            self.node_0.storage[f"{self.name}:z0"] = self.node_0.storage[f"{self.sub_protocol.name}:z0"] + \
                self.f_mul(self.node_0.storage[f"{self.sub_protocol.name}:x"], self.node_0.storage[f"{self.name}:y0"])
        
        if self.node_1.local():
            self.node_1.storage[f"{self.name}:z1"] = self.node_1.storage[f"{self.sub_protocol.name}:z1"]

        self.sub_protocol.clear_io()

    def clear_io(self):
        if self.node_0.local():
            del self.node_0.storage[f"{self.name}:y0"], self.node_0.storage[f"{self.name}:z0"]
        
        if self.node_1.local():
            del self.node_1.storage[f"{self.name}:z1"]


class SS_Perm(Protocol):
    def __init__(self, f_perm: Callable, 
                name: str, 
                node_0: Node, node_1: Node, node_2: Node,
                mask_scale: float, device: str="cpu") -> None:
        """
        To permute (or any operations that will modify the position of elements) a shared tensor
        The algorithm here used is permute + share
        See https://eprint.iacr.org/2019/1340.pdf for details
        """
        self.f_perm = f_perm
        self.name = name
        self.node_0 = node_0
        self.node_1 = node_1
        self.node_2 = node_2

        self.mask_scale = mask_scale
        self.device = device

    def prepare(self):
        pass

    def offline_execute(self, x_shape):
        """
        Input:
            Node 0: new_perm
        """

        # In node_0
        if self.node_0.local():
            self.node_0.send(self.node_2.name, f"{self.name}:new_perm", self.node_0.storage[f"{self.name}:new_perm"])
            self.node_0.enqueue(f"{self.name}:perm", self.node_0.storage.pop(f"{self.name}:new_perm"))

        # In node_2
        if self.node_2.local():
            perm = self.node_2.fetch(self.node_0.name, f"{self.name}:new_perm")
            mask_a = torch.rand(*x_shape, device=self.device) * self.mask_scale - self.mask_scale / 2
            mask_b = torch.rand(*x_shape, device=self.device) * self.mask_scale - self.mask_scale / 2
            perm_diff = self.f_perm(mask_a, perm) - mask_b
            
            self.node_2.send(self.node_0.name, f"{self.name}:perm_diff", perm_diff)
            self.node_2.send(self.node_1.name, f"{self.name}:mask_a&b", [mask_a, mask_b])

            del perm, mask_a, mask_b, perm_diff

        # In node_0
        if self.node_0.local():
            self.node_0.fetch_and_enqueue(self.node_2.name, f"{self.name}:perm_diff")
        
        if self.node_1.local():
            self.node_1.fetch_and_enqueue(self.node_2.name, f"{self.name}:mask_a&b")

    def online_execute(self):
        """
        Input:
            Node 0: x0
        Output: x1
            Node 0: z0
            Node 1: z1
        """
        # In node_1
        if self.node_1.local():
            mask_a, mask_b = self.node_1.storage[f"{self.name}:mask_a&b"].pop()
            self.node_1.storage[f"{self.name}:z1"] = mask_b
            self.node_1.send(self.node_0.name, f"{self.name}:x1-mask_a", self.node_1.storage[f"{self.name}:x1"] - mask_a)

        # In node_0
        if self.node_0.local():
            perm = self.node_0.storage[f"{self.name}:perm"].pop()
            self.node_0.storage[f"{self.name}:z0"] = \
                self.f_perm(self.node_0.storage[f"{self.name}:x0"], perm) + \
                self.node_0.storage[f"{self.name}:perm_diff"].pop() + \
                self.f_perm(self.node_0.fetch(self.node_1.name, f"{self.name}:x1-mask_a"), perm)

    
    def clear_io(self):
        if self.node_0.local():
            del self.node_0.storage[f"{self.name}:x0"], self.node_0.storage[f"{self.name}:z0"]
        
        if self.node_1.local():
            del self.node_1.storage[f"{self.name}:x1"], self.node_1.storage[f"{self.name}:z1"]




if __name__ == "__main__":
    @test_func
    def test__SS_Mul__CX_N0_Y_N1():
        print()
        communication = SimulatedCommunication(["n0", "n1", "n2"])
        communication.new_stage("Test")

        n0 = Node(communication, "n0")
        n1 = Node(communication, "n1")
        n2 = Node(communication, "n2")
        protocol_name = "ss_mul__cx_n0_y_n1"

        x = torch.tensor([[1, 2]]).float()
        y = torch.tensor([[2], [1]]).float()

        n0.storage[f"{protocol_name}:x"] = x
        n1.storage[f"{protocol_name}:y"] = y

        protocol = SS_Mul__CX_N0_Y_N1([1, 2], torch.matmul, protocol_name, n0, n1, n2, 10)
        protocol.prepare()
        protocol.offline_execute([2, 1], [1, 1])
        protocol.online_execute()



        print("-------------Output Expected/Executed--------------")
        print(x @ y)
        print(n0.storage[f"{protocol_name}:z0"] + n1.storage[f"{protocol_name}:z1"])


        print("----------Storage----------")
        print(n0.storage)
        print(n1.storage)    
        print("----------Storage (after clear IO)-----------")
        protocol.clear_io()
        print(n0.storage)
        print(n1.storage)    
        

    @test_func
    def test__SS_Mul__CX_N0():
        communication = SimulatedCommunication(["n0", "n1", "n2"])
        communication.new_stage("Test")

        n0 = Node(communication, "n0")
        n1 = Node(communication, "n1")
        n2 = Node(communication, "n2")

        protocol_name = "ss_mul__cx_n0"
        x = torch.tensor([[1, 2]]).float()
        y = torch.tensor([[1], [2]]).float()
        y0 = torch.tensor([[0.5], [3]]).float()
        n0.storage[f"{protocol_name}:x"] = x
        n0.storage[f"{protocol_name}:y0"] = y0
        n1.storage[f"{protocol_name}:y1"] = y - y0

        protocol = SS_Mul__CX_N0([1, 2], torch.matmul, protocol_name, n0, n1, n2, 10)
        protocol.prepare()
        protocol.offline_execute([2, 1], [1, 1])
        protocol.online_execute()

        print("-------------Output Expected/Executed--------------")
        print(x @ y)
        print(n0.storage[f"{protocol_name}:z0"] + n1.storage[f"{protocol_name}:z1"])


        print("----------Storage----------")
        print(n0.storage)
        print(n1.storage)    
        print("----------Storage (after clear IO)-----------")
        protocol.clear_io()
        print(n0.storage)
        print(n1.storage)    


    @test_func
    def test__SS_Perm():
        communication = SimulatedCommunication(["n0", "n1", "n2"])
        communication.new_stage("Test")

        n0 = Node(communication, "n0")
        n1 = Node(communication, "n1")
        n2 = Node(communication, "n2")

        protocol_name = "ss_perm"
        x = torch.tensor([1, 2, 3]).float()
        x0 = torch.rand_like(x)
        perm = torch.tensor([2, 1, 0])
        n0.storage[f"{protocol_name}:x0"] = x0
        n0.storage[f"{protocol_name}:new_perm"] = perm
        n1.storage[f"{protocol_name}:x1"] = x - x0

        f_perm = lambda x, perm: x[perm]

        protocol = SS_Perm(f_perm, protocol_name, n0, n1, n2, 10)
        protocol.prepare()
        protocol.offline_execute([3])
        protocol.online_execute()

        print("-------------Output Expected/Executed--------------")
        print(x[perm])
        print(n0.storage[f"{protocol_name}:z0"] + n1.storage[f"{protocol_name}:z1"])

        print("----------Storage----------")
        print(n0.storage)
        print(n1.storage)    
        print("----------Storage (after clear IO)-----------")
        protocol.clear_io()
        print(n0.storage)
        print(n1.storage)    



    test__SS_Mul__CX_N0_Y_N1()
    test__SS_Mul__CX_N0()
    test__SS_Perm()