#!/usr/bin/env python3

import basic_pb2
import basic_pb2_grpc
import grpc
import pickle
import numpy as np
import os

import threading

import cli_colors
import time

from concurrent import futures

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


class BasicServiceServicer(basic_pb2_grpc.BasicServiceServicer):
    model = None
    def __init__(self, backend, model_path, inputs, outputs) -> None:
        self.model = backend.load(model_path, inputs=inputs, outputs=outputs)
        super().__init__()

    def InferenceItem(self, request: basic_pb2.RequestItem, context: grpc.ServicerContext):
        items = pickle.loads(request.items)
        s = time.time()
        results = self.model.predict({self.model.inputs[0]: items})
        e = time.time()
        results = pickle.dumps((results, e-s), protocol=0)
        response: basic_pb2.ItemResult = basic_pb2.ItemResult(results=results)
        return response


def serve():
    model_path = os.path.join(os.environ["MODEL_DIR"], "mobilenet_v1_1.0_224.onnx")
    backend = get_backend("onnxruntime")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    basic_pb2_grpc.add_BasicServiceServicer_to_server(
        BasicServiceServicer(backend, model_path, None, ['MobilenetV1/Predictions/Reshape_1:0']), server)
    server.add_insecure_port('[::]:8086')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()