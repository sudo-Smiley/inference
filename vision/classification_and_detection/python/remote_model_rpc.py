#!/usr/bin/env python3

import basic_pb2
import basic_pb2_grpc
import grpc
import pickle
import numpy as np
import os

import argparse

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
    def __init__(self, backend, model_path, inputs, outputs, threads=0) -> None:
        self.model = backend.load(model_path, inputs=inputs, outputs=outputs, threads=threads)
        self.model_path = model_path
        self.backend = backend
        self.outputs = outputs
        self.inputs = inputs
        self.threads = threads
        super().__init__()

    def InferenceItem(self, request: basic_pb2.RequestItem, context: grpc.ServicerContext):
        items = pickle.loads(request.items)
        s = time.time()
        results = self.model.predict({self.model.inputs[0]: items})
        e = time.time()
        results = pickle.dumps((results, e-s), protocol=0)
        response: basic_pb2.ItemResult = basic_pb2.ItemResult(results=results)
        return response
    
    def ChangeThreads(self, request, context):
        n = request.threads
        if n == self.threads:
            cli_colors.color_print("Request to change threads ignored", cli_colors.YELLOW)
            return basic_pb2.ThreadReply(ok=True)
        self.model = self.backend.load(self.model_path, self.inputs, self.outputs, n)
        
        self.threads = n
        return basic_pb2.ThreadReply(ok=True)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-threads", type=int, default=0,
                        help="the number of threads the model should run for inferencing a single query")
    args = parser.parse_args()
    return args

def serve():
    args = get_args()
    model_path = os.path.join(os.environ["MODEL_DIR"], "ssd_mobilenet_v1_coco_2018_01_28.onnx")
    backend = get_backend("onnxruntime")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    basic_pb2_grpc.add_BasicServiceServicer_to_server(
        BasicServiceServicer(backend, model_path, None, ['num_detections:0','detection_boxes:0','detection_scores:0','detection_classes:0'], threads=args.model_threads), server)
    server.add_insecure_port('localhost:8086')
    server.start()
    server.wait_for_termination()



if __name__ == '__main__':
    serve()