import torch
from torch import nn


from perm_llm.glm6b.secure_inference import GLM_PredictionProtocol, GLM_EmbeddingRetrievalProtocol
from perm_llm.common.torch_utils import relative_error
from perm_llm.common.communication import Communication, Node, SimulatedCommunication
from perm_llm.common.utils import test_func



prediction_layer = nn.Linear(4096, 130528, bias=False).cuda()


hs = prediction_layer.weight[130527]
h0 = torch.normal(0, 10, [4096]).cuda()
h1 = hs - h0

prediction = torch.argmax(prediction_layer(hs))
print("Expected prediction: ", prediction.item())

communication = SimulatedCommunication(["n0", "n1", "n2"])
communication.new_stage("Test")

n0 = Node(communication, "n0")
n1 = Node(communication, "n1")
n2 = Node(communication, "n2")

n0.space.word_embedding = prediction_layer.weight

embedding_protocol = GLM_EmbeddingRetrievalProtocol(n0, n1, n2, 10, device="cuda")
prediction_protocol = GLM_PredictionProtocol(n0, n1, n2, embedding_protocol.name, 10, device="cuda")
embedding_protocol.prepare()
prediction_protocol.prepare()
prediction_protocol.offline_execute()

n0.storage[f"{prediction_protocol.name}:x0"] = h0
n1.storage[f"{prediction_protocol.name}:x1"] = h1
prediction_protocol.online_execute()
print("Computed prediction: ", n1.storage[f"{prediction_protocol.name}:z"])