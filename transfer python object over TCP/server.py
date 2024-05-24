import socket
import pickle
import cv2

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server.bind(('127.0.0.1', 9999))

server.listen(1)

while True:
    print("Waiting for connection...")
    client, addr = server.accept()

    try:
        print("Connected")

        data = b''
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk
        
        recieved_object = pickle.loads(data)
        print(f"Recieved data: {recieved_object}")
        cv2.imwrite("transfared.jpg", recieved_object)
    finally:
        client.close()