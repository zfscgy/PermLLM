import time
import threading

from simple_socket.zf_socket import SocketServer


if __name__ == "__main__":
    server0 = SocketServer("127.0.0.1:12000", {"127.0.0.1:12001": "p1", "127.0.0.1:12002": "p2"})
    server1 = SocketServer("127.0.0.1:12001", {"127.0.0.1:12000": "p0", "127.0.0.1:12002": "p2"})
    server2 = SocketServer("127.0.0.1:12002", {"127.0.0.1:12000": "p0", "127.0.0.1:12001": "p1"})
    print("Server 0 & 1 & 2 initialized, start listening...")
    time.sleep(1)
    server0.connect_all()
    print("Server 0 connection complete.")
    server1.connect_all()
    print("Server 1 connection complete.")
    server2.connect_all()
    print("Server 2 connection complete.")


    th_server1 = threading.Thread(target=server1.send_to, args=("p0", b"1" * 100000))    
    th_server2 = threading.Thread(target=server2.send_to, args=("p0", b"0" * 100000))    

    th_server1.start()
    th_server2.start()

    received1 = server0.recv_from("p1")
    received2 = server0.recv_from("p2")
    print(f"Server 1's message: {received1}")
    print(f"Server 2's message: {received2}")

    th_server1.join()
    th_server2.join()