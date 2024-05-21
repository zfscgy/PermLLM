## Dependencies
You may only install the following libraries mannually with Conda to run PermLLM

*   Basic libraries
    ```
    pytorch (newest version is ok I guess)
    ```

*   Required by ChatGLM-6B
    ```
    transformers=4.27.1  (APIs change often, so other versions don't work!)
    sentencepiece
    ```
    To check, run `llm_bases/chatglm6b.py`

*   Required by PermLLM
    ```
    Pyfhel  (For BFV homomorphic encryption)
    ```

## To reproduce the results

ChatGLM-6B model with PermLLM: 

```
perm_llm\glm6b\test\secure_inference_test__multiparty_whole_processing.py
```

Benchmark for one single transformer layer:

```
perm_llm\glm6b\test\secure_inference_test__multiparty_layer_processing.py
```

The file shall be run in 3 terminals at the same time, to simulate $P_0, P_1$ and $P_2$. Before running, add the project root to the `PYTHONPATH` environment variable.

### Network simulation

We suggest to use the `tcconfig` (pypi) lib for convenient network simulation, which is a wrapper for the linux `tc` (traffic control) command.

Use `ping 127.0.0.1` or `ping localhost` to verify that the simulation is successful before perform experiments.

However,  `tc` requires the root privilege. You can also simulate the network by

`comm.simulate_network(delay_ms, bandiwdth_Mbps)`

It can yield similar results with `tc`



# API Reference

## Communication

The `Communication` class is a abstract communication layer. It implements the `send` and `receive` method between different nodes (parties). The `send` method is asynchronous, while the receive `method` is **synchronous** (blocking).

The base `Communication` class does not consider threading. When using it, all the nodes shall be attached to the same `Communication` instance. The `RealCommunication` class is based on the socket.

## Node

A `Node` is a party participating in the secure computation protocols. 

It need a `Communication` instance to be initialized.

It has a `storage` field to store the variables produced and needed in the execution of protocols.

## Protocol

The `Protocol` class take nodes for initialization. When perform actual multiparty computation, there will be dummy nodes,  i,e, the node does not exist on the current machine. The code related to the dummy code will not be executed. The dummy nodes only serves as a name for `send` and `receive`. Also, dummy nodes do not have the `sotrage` domain.

For example, consider a Protocol with three parties, and will be executed on 3 machines. Then the protocol definition in three machines will be

```python
protocol = Protocol(node0, Dummy_node1, Dummy_node2)  # In machine 0
#-----------------------------------------------------
protocol = Protocol(Dummy_node0, node1, Dummy_node2)  # In machine 1
#-----------------------------------------------------
protocol = Protocol(Dummy_node0, Dummy_node1, node2)  # In machine 2
```
