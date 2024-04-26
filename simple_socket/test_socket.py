import time
import threading

from simple_socket.zf_socket import SocketServer


if __name__ == "__main__":
    server0 = SocketServer("127.0.0.1:9001", {"127.0.0.1:9002": "p1"})
    server1 = SocketServer("127.0.0.1:9002", {"127.0.0.1:9001": "p0"})
    print("Server 0 & 1 initialized, start listening...")
    time.sleep(1)
    server0.connect_all()
    print("Server 0 connection complete.")
    server1.connect_all()
    print("Server 1 connection complete.")

    server0.send_to("p1", b"Hello from server0")
    print("Server 0 sent.")

    received = server1.recv_from("p0")
    print(received)