import array
import time
import numpy as np
import mlperf_loadgen as lg

import socket
import cli_colors
import pickle

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id, img, label=None):
        self.query_id = query_id
        self.content_id = content_id
        self.img: np.ndarray = img
        self.label = label
        self.start = time.time()


class RemoteRunnerBase:
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        self.take_accuracy = False
        self.ds = ds
        # self.model = model
        self.post_process = post_proc
        self.threads = threads
        self.take_accuracy = False
        self.max_batchsize = max_batchsize
        self.result_timing = []
        self.server = ("127.0.0.1", 8085)
        self.connect()

    def connect(self):
        self.sockfd = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.sockfd.connect(self.server)


    def handle_tasks(self, tasks_queue):
        pass

    def start_run(self, result_dict, take_accuracy):
        self.result_dict = result_dict
        self.result_timing = []
        self.take_accuracy = take_accuracy
        self.post_process.start()

    def predict_remote(self, qitem: Item):
        payload = pickle.dumps(qitem.img, protocol=0)
        size = len(payload)
        size_bytes = size.to_bytes(4, 'big')
        
        self.sockfd.send(size_bytes + payload)
        res_size = self.sockfd.recv(4, socket.MSG_WAITALL)
        res = self.sockfd.recv(int.from_bytes(res_size, 'big'), socket.MSG_WAITALL)
        return pickle.loads(res)


    def run_one_item(self, qitem):
        # run the prediction
        processed_results = []
        try:
            # results = self.model.predict({self.model.inputs[0]: qitem.img})
            results = self.predict_remote(qitem)
            processed_results = self.post_process(results, qitem.content_id, qitem.label, self.result_dict)
            if self.take_accuracy:
                self.post_process.add_results(processed_results)
                self.result_timing.append(time.time() - qitem.start)
        except Exception as ex:  # pylint: disable=broad-except
            src = [self.ds.get_item_loc(i) for i in qitem.content_id]
            log.error("thread: failed on contentid=%s, %s", src, ex)
            # since post_process will not run, fake empty responses
            processed_results = [[]] * len(qitem.query_id)
        finally:
            response_array_refs = []
            response = []
            for idx, query_id in enumerate(qitem.query_id):
                response_array = array.array("B", np.array(processed_results[idx], np.float32).tobytes())
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
            lg.QuerySamplesComplete(response)

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        if len(query_samples) < self.max_batchsize:
            data, label = self.ds.get_samples(idx)
            self.run_one_item(Item(query_id, idx, data, label))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                data, label = self.ds.get_samples(idx[i:i+bs])
                self.run_one_item(Item(query_id[i:i+bs], idx[i:i+bs], data, label))

    def finish(self):
        pass