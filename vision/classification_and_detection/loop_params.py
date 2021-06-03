#!/usr/bin/env python3

import subprocess
import cli_colors as c
import os
import tqdm
import python.basic_pb2
import python.basic_pb2_grpc
import grpc

SUT_address = "localhost:8086"

def change_model_threads(n):
    channel = grpc.insecure_channel(SUT_address)
    stub = python.basic_pb2_grpc.BasicServiceStub(channel)
    res: python.basic_pb2.ThreadReply = stub.ChangeThreads(python.basic_pb2.ThreadRequest(threads=n))
    if res.ok:
        c.color_print(f"Changed to {n} threads", c.CYAN)
    else:
        c.color_print(f"Changing to {n} threads failed", c.RED)

def main():
    path = "output/onnxruntime-cpu/ssd-mobilenet/"
    spq_max = 2
    clients_max = 2
    threads = [0, 1, 2, 4, 8, 16, 24, 32]
    command = f"./run_remote.sh onnxruntime ssd-mobilenet cpu --scenario MultiStream --time 10  --address {SUT_address} --qps 400".split()
    command_offline = f"./run_remote.sh onnxruntime ssd-mobilenet cpu --scenario Offline --address {SUT_address} --count 1000".split()
    for n in threads:
        change_model_threads(n)
        for spq in tqdm.tqdm(range(1, spq_max), desc="SPQ", leave=False):
            for clients in tqdm.tqdm(range(1, clients_max), leave=False, desc="Clients"):
                new_command = command_offline + ["--threads", str(clients), "--max-batchsize", str(spq)]
                subprocess.run(" ".join(new_command), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                d_path = os.path.join(path, f"{clients}-{spq}-{0}")
                try:
                    os.mkdir(d_path)
                except:
                    pass
                files = os.listdir(path)
                for f in files:
                    f_path = os.path.join(path, f)
                    f_new_path = os.path.join(d_path, f)
                    if os.path.isfile(f_path):
                        os.rename(f_path, f_new_path)

if __name__ == "__main__":
    main()