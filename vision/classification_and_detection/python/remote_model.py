#!/usr/bin/env python3


import pickle
import socket
import cli_colors
import numpy as np
import os

class ModelServer():

    def __init__(self, backend, model_path, inputs, outputs):
        self.model = backend.load(model_path, inputs=inputs, outputs=outputs)
        self.listen()

    def listen(self):
        self.sockfd = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.sockfd.bind(("0.0.0.0", 8085))
        print(f"Waiting")
        self.sockfd.listen()
        self.clifd, addr = self.sockfd.accept()
        # self.clifd.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)
        # print(self.clifd.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        print(f"connected to {addr}") 

    def get_item(self):
        # cli_colors.color_print("get_item", cli_colors.YELLOW)
        item_size = self.clifd.recv(4)
        if len(item_size) == 0:
            cli_colors.color_print("LoadGen Disconnected", cli_colors.CYAN)
            return False
        item_size = int.from_bytes(item_size, "big")
        item = self.clifd.recv(item_size, socket.MSG_WAITALL)
        item: np.ndarray = pickle.loads(item)
        results = self.model.predict({self.model.inputs[0]: item})
        # cli_colors.color_print(results, cli_colors.GREEN)
        results = pickle.dumps(results, protocol=0)
        results_size = len(results).to_bytes(4, "big")
        self.clifd.send(results_size + results)
        return True

def get_backend(backend):
    if backend == "tensorflow":
        from backend_tf import BackendTensorflow
        backend = BackendTensorflow()
    elif backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime
        backend = BackendOnnxruntime()
    elif backend == "null":
        from backend_null import BackendNull
        backend = BackendNull()
    elif backend == "pytorch":
        from backend_pytorch import BackendPytorch
        backend = BackendPytorch()
    elif backend == "pytorch-native":
        from backend_pytorch_native import BackendPytorchNative
        backend = BackendPytorchNative()
    elif backend == "tflite":
        from backend_tflite import BackendTflite
        backend = BackendTflite()
    else:
        raise ValueError("unknown backend: " + backend)
    return backend


def main():


    model_path = os.environ["MODEL_DIR"]
    backend = get_backend("onnxruntime")
    model_server = ModelServer(
        backend, model_path, None, ['MobilenetV1/Predictions/Reshape_1:0'])
    while 1:
        if model_server.get_item() == False:
            break





if __name__ == "__main__":
    main()
