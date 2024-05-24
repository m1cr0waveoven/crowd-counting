import socket
import pickle
import cv2

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 9999))

try:
    myobject = { 'key1': 'value1', 'key2': 'value2'}
    img = cv2.imread("known_face.jpg")
    serialized = pickle.dumps(img)
    client.sendall(serialized)
finally:
    client.close()
