"""Bridges stdin/stdout <-> 127.0.0.1:9000 inside the pod. Run once per connection through
`kubectl exec -i`; it copies bytes and does nothing else.
"""
import os
import socket
import sys
import threading
import time

HOST, PORT = "127.0.0.1", 9000

# Frames go out on a private dup of fd 1; fd 1 itself is pointed at stderr so that nothing
# a library prints can end up inside the binary stream.
OUT_FD = os.dup(1)
os.dup2(2, 1)


def connect(timeout=10.0):
    deadline = time.time() + timeout
    while True:
        try:
            sock = socket.create_connection((HOST, PORT), timeout=5)
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.2)


def stdin_to_socket(sock):
    try:
        while True:
            data = os.read(0, 1 << 16)
            if not data:
                break
            sock.sendall(data)
    except OSError:
        pass
    # EOF from the client: closing the socket also ends the other direction
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def socket_to_stdout(sock):
    try:
        while True:
            data = sock.recv(1 << 16)
            if not data:
                break
            view = memoryview(data)
            while view:
                view = view[os.write(OUT_FD, view):]
    except OSError:
        pass


def main():
    try:
        sock = connect()
    except OSError as e:
        print(f"relay: cannot reach {HOST}:{PORT}: {e}", file=sys.stderr)
        return 1
    threading.Thread(target=stdin_to_socket, args=(sock,), daemon=True).start()
    socket_to_stdout(sock)
    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
