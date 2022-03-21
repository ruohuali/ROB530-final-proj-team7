import socket
import struct

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 8080  # Port to listen on (non-privileged ports are > 1023)

def pseudoFloatList(length):
    return [x / 0.3 for x in range(length)]

def floatList2Bytes(lst):
    buf = bytes()
    for val in lst:
        buf += struct.pack('d', val)
    return buf    

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(200)
            if not data:
                break
            print("len", len(bytearray(data)), len(data))
            for d in bytearray(data):
                print(d)
            print("all char received")
            lst = pseudoFloatList(7)
            msg = floatList2Bytes(lst)
            conn.sendall(msg)
            print("all float sent")