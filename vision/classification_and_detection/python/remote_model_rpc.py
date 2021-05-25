#!/usr/bin/env python3

import basic_pb2
import basic_pb2_grpc
import grpc
import pickle
import numpy as np
import os

import cli_colors

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

    def InferenceItem(self, request: basic_pb2.RequestItem, context):
        cli_colors.color_print("Inference Item", cli_colors.CYAN_SHADE2)
        items = pickle.loads(request.items)
        results = self.model.predict({self.model.inputs[0]: items})
        results = pickle.dumps(results, protocol=0)
        response: basic_pb2.ItemResult = basic_pb2.ItemResult(results=results)
        cli_colors.color_print("Ended inferencing", cli_colors.CYAN_SHADE1)
        return response


def serve():
    model_path = os.environ["MODEL_DIR"]
    backend = get_backend("onnxruntime")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    basic_pb2_grpc.add_BasicServiceServicer_to_server(
        BasicServiceServicer(backend, model_path, None, ['MobilenetV1/Predictions/Reshape_1:0']), server)
    server.add_insecure_port('[::]:8086')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()