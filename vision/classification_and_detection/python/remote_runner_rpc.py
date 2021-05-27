import array
import time
import numpy as np
import mlperf_loadgen as lg
import threading
from queue import Queue
import math
import basic_pb2
import basic_pb2_grpc
import grpc


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

    time_taken = []
    pickling = []
    coms = []
    batches = 0
    samples = 0
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        self.take_accuracy = False
        self.ds = ds
        # self.model = model
        self.post_process = post_proc
        self.threads = threads
        self.take_accuracy = False
        self.max_batchsize = max_batchsize
        self.result_timing = []
        self.connect()
    

    def connect(self):
        self.channel = grpc.insecure_channel('localhost:8086')
        self.stub = basic_pb2_grpc.BasicServiceStub(self.channel)

    def request_model(self, model):
        pass

    def handle_tasks(self, tasks_queue):
        pass

    def start_run(self, result_dict, take_accuracy):
        self.result_dict = result_dict
        self.result_timing = []
        self.take_accuracy = take_accuracy
        self.post_process.start()

    def predict_remote(self, qitem: Item):
        s = time.time()
        item_pickle = pickle.dumps(qitem.img)
        p1 = time.time()
        request = basic_pb2.RequestItem(items=item_pickle)
        response: basic_pb2.ItemResult = self.stub.InferenceItem(request)
        p2 = time.time
        result, time_taken = pickle.loads(response.results)
        e = time.time()
        self.time_taken.append(time_taken)
        self.coms.append(p2 - p1 - time_taken)
        self.pickling.append((p1 - s, e - p2))
        return result


    def run_one_item(self, qitem):
        # run the prediction
        processed_results = []
        try:
            # results = self.model.predict({self.model.inputs[0]: qitem.img})
            cli_colors.color_print("Calling Predict", cli_colors.BLUE_SHADE1)
            results = self.predict_remote(qitem)
            cli_colors.color_print("Predict Returned", cli_colors.RED_SHADE1)
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
        self.samples += len(query_samples)
        self.batches += math.ceil(len(query_samples)/self.max_batchsize)
        if len(query_samples) < self.max_batchsize:
            data, label = self.ds.get_samples(idx)
            self.run_one_item(Item(query_id, idx, data, label))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                data, label = self.ds.get_samples(idx[i:i+bs])
                self.run_one_item(Item(query_id[i:i+bs], idx[i:i+bs], data, label))

    def finish(self):
        cli_colors.color_print(f"Total: {(sum(self.time_taken))}, Avg: {(sum(self.time_taken)/len(self.time_taken)):5f}", cli_colors.RED)
        cli_colors.color_print(f"{self.samples} Samples in {self.batches} Batches", cli_colors.GREEN)
        import pandas as pd
        p_df = pd.DataFrame(self.pickling, columns=["Pickle", "Unpickle"])
        c_df = pd.DataFrame(self.coms)
        t_df = pd.DataFrame(self.time_taken)
        p_df.to_csv("pickling.csv")
        c_df.to_csv("communication.csv")
        t_df.to_csv("inference.csv")
        pass

class RemoteQueueRunner(RemoteRunnerBase):
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        super().__init__(model, ds, threads, post_proc, max_batchsize)
        self.tasks = Queue(maxsize=threads * 4)
        self.workers = []
        self.result_dict = {}

        for _ in range(self.threads):
            worker = threading.Thread(target=self.handle_tasks, args=(self.tasks,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()

    def handle_tasks(self, tasks_queue):
        """Worker thread."""
        while True:
            qitem = tasks_queue.get()
            if qitem is None:
                # None in the queue indicates the parent want us to exit
                tasks_queue.task_done()
                break
            self.run_one_item(qitem)
            tasks_queue.task_done()

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        self.samples += len(query_samples)
        self.batches += math.ceil(len(query_samples)/self.max_batchsize)
        if len(query_samples) < self.max_batchsize:
            data, label = self.ds.get_samples(idx)
            self.tasks.put(Item(query_id, idx, data, label))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                ie = i + bs
                data, label = self.ds.get_samples(idx[i:ie])
                self.tasks.put(Item(query_id[i:ie], idx[i:ie], data, label))

    def finish(self):
        # exit all threads
        for _ in self.workers:
            self.tasks.put(None)
        for worker in self.workers:
            worker.join()
        super().finish()