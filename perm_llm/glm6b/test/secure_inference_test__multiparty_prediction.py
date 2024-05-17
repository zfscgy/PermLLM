# %%
import time

import threading

from simple_socket.zf_socket import SocketServer
from perm_llm.common.communication import Node
from perm_llm.common.communication import Communication, Node, SimulatedCommunication
from perm_llm.common.real_communication import RealCommunication

device = "cuda"


# Set up communication

address_dict = {
    "127.0.0.1:9100": "n0",
    "127.0.0.1:9101": "n1",
    "127.0.0.1:9102": "n2"
}
sock0 = SocketServer("127.0.0.1:9100", address_dict, 1000)
sock1 = SocketServer("127.0.0.1:9101", address_dict, 1000)
sock2 = SocketServer("127.0.0.1:9102", address_dict, 1000)

time.sleep(1) # Wait the server to start listening


connect1 = threading.Thread(target=sock1.connect_all)
connect2 = threading.Thread(target=sock2.connect_all)
connect1.start()
connect2.start()
sock0.connect_all()
connect1.join()
connect2.join()
print("All sockets connected")

comm0 = RealCommunication({"n0": sock0}, tensor_device=device)
comm1 = RealCommunication({"n1": sock1}, tensor_device=device)
comm2 = RealCommunication({"n2": sock2}, tensor_device=device)
n0 = Node(comm0, "n0")
n1 = Node(comm1, "n1")
n2 = Node(comm2, "n2")


import torch
from perm_llm.glm6b.secure_inference import GLMConfig

n0.space.word_embedding = torch.normal(0, 1, [GLMConfig.n_tokens, 4096]).cuda()

hs = torch.normal(0, 1, [4096]).cuda()
h0 = torch.normal(0, 10, [4096]).cuda()
h1 = hs - h0


prediction = torch.argmax(n0.space.word_embedding @ hs)
print("Expected prediction: ", prediction.item())


from perm_llm.glm6b.secure_inference import GLM_PredictionProtocol, GLM_EmbeddingRetrievalProtocol, GLMConfig
from perm_llm.common.torch_utils import relative_error


embedding_protocol0 = GLM_EmbeddingRetrievalProtocol(n0, Node.from_remote_name("n1"), Node.from_remote_name("n2"), 10, device="cuda")
embedding_protocol1 = GLM_EmbeddingRetrievalProtocol(Node.from_remote_name("n0"), n1, Node.from_remote_name("n2"), 10, device="cuda")
embedding_protocol2 = GLM_EmbeddingRetrievalProtocol(Node.from_remote_name("n0"), Node.from_remote_name("n1"), n2, 10, device="cuda")

prediction_protocol0 = GLM_PredictionProtocol(n0, Node.from_remote_name("n1"), Node.from_remote_name("n2"), embedding_protocol0.name, 10, device="cuda")
prediction_protocol1 = GLM_PredictionProtocol(Node.from_remote_name("n0"), n1, Node.from_remote_name("n2"), embedding_protocol1.name, 10, device="cuda")
prediction_protocol2 = GLM_PredictionProtocol(Node.from_remote_name("n0"), Node.from_remote_name("n1"), n2, embedding_protocol2.name, 10, device="cuda")



# Prepare embedding_retrieval
emb_prepare1 = threading.Thread(target=embedding_protocol1.prepare)
emb_prepare2 = threading.Thread(target=embedding_protocol2.prepare)
emb_prepare1.start()
emb_prepare2.start()
embedding_protocol0.prepare()
emb_prepare1.join()
emb_prepare2.join()

print("embedding_retrieval prepare finished")

prepare1 = threading.Thread(target=prediction_protocol1.prepare)
prepare2 = threading.Thread(target=prediction_protocol2.prepare)
prepare1.start()
prepare2.start()
prediction_protocol0.prepare()
prepare1.join()
prepare2.join()

print("prediction prepare finished")

offline1 = threading.Thread(target=prediction_protocol1.offline_execute)
offline2 = threading.Thread(target=prediction_protocol2.offline_execute)
offline1.start()
offline2.start()
prediction_protocol0.offline_execute()
offline1.join()
offline2.join()



n0.storage[f"{prediction_protocol0.name}:x0"] = h0
n1.storage[f"{prediction_protocol1.name}:x1"] = h1
online1 = threading.Thread(target=prediction_protocol1.online_execute)
online2 = threading.Thread(target=prediction_protocol2.online_execute)
online1.start()
online2.start()
prediction_protocol0.online_execute()
online1.join()
online2.join()
print("Computed prediction: ", n1.storage[f"{prediction_protocol1.name}:z"])

n2.storage['GLM__EmbeddingLayer/onehot_matmul:beaver_u'].shape





