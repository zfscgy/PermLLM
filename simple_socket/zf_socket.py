from typing import Dict

import time
import socket
import threading
import logging

from io import BytesIO


logger = logging.getLogger("Socket")


class SocketConfig:
    start_flag = b'ZFCommProtocol v2024.04.25'
    len_header = 6
    buffer_size = 1024 * 1024


class SocketException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def read_socket(s: socket.socket) -> bytes:
    try:
        flag = s.recv(len(SocketConfig.start_flag))
        if flag != SocketConfig.start_flag:
            raise SocketException("Protocol not match")

        len_bytes = s.recv(SocketConfig.len_header)
        if len(len_bytes) == 0:
            raise SocketException("No data to read")

        content_len = int.from_bytes(len_bytes, byteorder='big')
        logger.debug("Get message size %d" % content_len)
        received_content = BytesIO()
        while received_content.getbuffer().nbytes < content_len:
            received_content.write(s.recv(min(SocketConfig.buffer_size, content_len - received_content.getbuffer().nbytes)))
        received_content.seek(0)
        return received_content.read()
    except Exception as e:
        raise SocketException(f"Socket read error: {e}")


def write_socket(s: socket.socket, content: bytes):
    try:
        content_len = len(content) + SocketConfig.len_header + len(SocketConfig.start_flag)
        len_bytes = len(content).to_bytes(SocketConfig.len_header, byteorder='big')
        send_bytes = SocketConfig.start_flag + len_bytes + content
        sent_len = 0
        while sent_len != content_len:
            end_pos = min(len(send_bytes), sent_len + SocketConfig.buffer_size)
            current_bytes = send_bytes[sent_len: end_pos]
            sent_len += s.send(current_bytes)

    except Exception as e:
        raise SocketException(f"Socket send error: {e}")


class SocketServer:
    def __init__(self, address: str, other_addrs: dict, timeout=100):
        """
        :param address:  IP:port format, e.g., 127.0.0.1:8888
        :param other_addrs: dict[address, name]
        :param timeout:
        """
        self.addr = address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            ipv4, port = address.split(":")
            port = int(port)

        except:
            raise SocketException("Address %s not valid" % address)

        # setsockopt should be called before binding the socket.
        # Use SO_REUSEPORT to prevent 'Address already in use' problem
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((ipv4, port))

        logger.debug("Default timeout set to %d" % timeout)
        socket.setdefaulttimeout(timeout)
        self.other_addrs = other_addrs
        self.other_recv_sockets: Dict[str, socket.socket] = dict()
        self.other_send_sockets: Dict[str, socket.socket] = dict()
        self.send_locks = dict()
        self.waiting_for_connection = True

        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.start()

        # Count traffic from/to
        self.traffic_counter_to = dict()
        self.traffic_counter_from = dict()

    def set_timeout(self, timeout: float):
        logger.debug("Timeout set to %d" % timeout)
        for sock in self.other_recv_sockets.values() + self.other_send_sockets.values():
            sock: socket.socket
            sock.settimeout(timeout)

    def _listen_loop(self):
        self.socket.listen()
        not_connected_others = set(self.other_addrs.keys())
        while self.waiting_for_connection:
            try:
                accpeted_socket, addr = self.socket.accept()
            except TimeoutError as e:
                continue

            try:
                claimed_addr = str(read_socket(accpeted_socket), "utf-8")
            except TimeoutError:
                raise SocketException("Did not receive address claim after connection from %s" % addr)

            if claimed_addr.split(":")[0] != addr[0]:
                raise SocketException("Claimed Address %s do not match with the actual send address %s"
                                      % (claimed_addr, addr[0]))
            if claimed_addr in self.other_addrs:
                self.other_recv_sockets[self.other_addrs[claimed_addr]] = accpeted_socket
                self.traffic_counter_from[self.other_addrs[claimed_addr]] = 0
            else:
                raise SocketException("Get unexpected socket connection from %s" % addr)

            not_connected_others.remove(claimed_addr)
            if len(not_connected_others) == 0:
                break
        self.waiting_for_connection = False

    def connect_all(self):
        def connect_one(peer_addr: str, peer_name: str):
            my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                peer_ipv4, peer_port = peer_addr.split(":")
                peer_port = int(peer_port)
            except:
                raise SocketException("%s is not a valid address" % peer_addr)

            try:
                my_socket.connect((peer_ipv4, peer_port))
                write_socket(my_socket, self.addr.encode("utf-8"))
            except TimeoutError:
                raise SocketException("Connect to %s: %s failed" % (peer_name, peer_addr))
            self.other_send_sockets[peer_name] = my_socket
            self.traffic_counter_to[peer_name] = 0
            self.send_locks[peer_name] = threading.Lock()

        peers = [(peer_addr, self.other_addrs[peer_addr]) for peer_addr in self.other_addrs]
        for peer in peers:
            connect_one(*peer)
        # while self.listening:
        #     time.sleep(0.1)
        return

    def send_to(self, name: str, data: bytes):
        self.send_locks[name].acquire()
        if name not in self.other_send_sockets:
            raise SocketException("Peer name %s dose not exist or not connected yet" % name)
        s = self.other_send_sockets[name]
        write_socket(s, data)

        msg_len = len(data) + SocketConfig.len_header + len(SocketConfig.start_flag)
        self.traffic_counter_to[name] += msg_len
        self.send_locks[name].release()


    def recv_from(self, name):
        """
        Notice: do not call this in parallel with the same name.
        """
        if name not in self.other_recv_sockets:
            raise SocketException("Peer name %s dose not exist or not connected yet" % name)
        s = self.other_recv_sockets[name]
        content = read_socket(s)
        self.traffic_counter_from[name] += len(content) + SocketConfig.len_header + len(SocketConfig.start_flag)
        return content

    def reset_counter(self):
        for k in self.traffic_counter_from:
            self.traffic_counter_from[k] = 0
        for k in self.traffic_counter_to:
            self.traffic_counter_to[k] = 0

    def terminate(self):
        self.socket.close()
        for peer_name in self.other_send_sockets:
            self.other_send_sockets[peer_name].close()
        for peer_name in self.other_recv_sockets:
            self.other_recv_sockets[peer_name].close()